"""
FastAPI-сервис Sentinel-Q.

Эндпоинты:
    GET  /healthz                     — проверка живости
    GET  /v1/profile                  — метаданные профиля
    POST /v1/detect                   — детекция одного образца
    POST /v1/detect/batch             — пакетная детекция

Формат запроса /v1/detect:
    {
      "sample_b64": "base64-encoded bytes",
      "predicted_label": 3,
      "combine": "weighted",     // опционально
      "threshold": 1.5            // опционально
    }

Формат ответа:
    {
      "is_adversarial": true,
      "score": 3.01,
      "z_frag": +4.20,
      "z_H": +2.80,
      "z_unique": +0.05,
      "predicted_label": 3,
      "reason": "frag z=+4.20, H z=+2.80, n_unique z=+0.05"
    }

Запуск:
    sentinel-q serve --profile profile_mnist.json --host 0.0.0.0 --port 8000
    # или напрямую:
    uvicorn sentinel_q.server:create_app --factory --host 0.0.0.0 --port 8000

Требует:
    pip install sentinel-q[server]
"""

from __future__ import annotations
import base64
import os
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "Установите серверные зависимости: pip install sentinel-q[server]"
    )

from .profile import ReferenceProfile
from .detector import SentinelDetector


# ── Pydantic-схемы ───────────────────────────────────────────

class DetectRequest(BaseModel):
    sample_b64: str = Field(..., description="base64-encoded bytes")
    predicted_label: int = Field(..., ge=0, description="класс, к которому отнесён образец")
    combine: Optional[str] = Field(None, description="weighted | frag_only | mean | max")
    threshold: Optional[float] = Field(None, ge=0.0, description="порог score")


class DetectResponse(BaseModel):
    is_adversarial: bool
    score: float
    z_frag: float
    z_H: float
    z_unique: float
    predicted_label: int
    reason: str


class BatchDetectRequest(BaseModel):
    samples_b64: list[str]
    predicted_labels: list[int]
    combine: Optional[str] = None
    threshold: Optional[float] = None


class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]
    n_adversarial: int
    n_total: int
    mean_score: float


class ProfileInfo(BaseModel):
    n_samples: int
    n_classes: int
    label_space: list[int]
    feature_names: list[str]
    stats: dict


# ── Фабрика приложения ──────────────────────────────────────

def create_app(profile_path: str | None = None,
               default_combine: str = "weighted",
               default_threshold: float = 1.5) -> FastAPI:
    """
    Создаёт FastAPI-приложение.

    profile_path берётся из:
      1. Аргумента функции
      2. Переменной окружения SENTINEL_Q_PROFILE
    """
    profile_path = profile_path or os.environ.get("SENTINEL_Q_PROFILE")
    if not profile_path:
        raise RuntimeError(
            "Не задан profile_path. Передайте в create_app() или "
            "установите SENTINEL_Q_PROFILE=/path/to/profile.json"
        )

    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Профиль не найден: {profile_path}")

    profile = ReferenceProfile.load(profile_path)
    detector = SentinelDetector(
        profile,
        z_threshold=default_threshold,
        combine=default_combine,
    )

    app = FastAPI(
        title="Sentinel-Q",
        description="Model-free adversarial detection service",
        version="0.1.0",
    )

    # ── Health check ─────────────────────────────────────
    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "profile": str(profile_path),
            "n_samples": profile.n_samples_total,
            "n_classes": len(profile.stats),
            "default_combine": default_combine,
            "default_threshold": default_threshold,
        }

    # ── Метаданные профиля ───────────────────────────────
    @app.get("/v1/profile", response_model=ProfileInfo)
    def get_profile():
        return ProfileInfo(
            n_samples=profile.n_samples_total,
            n_classes=len(profile.stats),
            label_space=[int(x) for x in profile.label_space],
            feature_names=profile.feature_names,
            stats={str(k): v for k, v in profile.stats.items()},
        )

    # ── Один образец ─────────────────────────────────────
    @app.post("/v1/detect", response_model=DetectResponse)
    def detect(req: DetectRequest):
        try:
            raw = base64.b64decode(req.sample_b64, validate=True)
        except Exception as e:
            raise HTTPException(400, f"invalid base64: {e}")

        combine = req.combine or default_combine
        threshold = req.threshold if req.threshold is not None else default_threshold

        # Переиспользуем detector, но с локальными параметрами если нужно
        if combine != detector.combine or threshold != detector.z_threshold:
            local_detector = SentinelDetector(
                profile, z_threshold=threshold, combine=combine
            )
        else:
            local_detector = detector

        v = local_detector.check(raw, req.predicted_label)
        return DetectResponse(
            is_adversarial=v.is_adversarial,
            score=v.score,
            z_frag=v.z_frag,
            z_H=v.z_H,
            z_unique=v.z_unique,
            predicted_label=v.predicted_label,
            reason=v.reason,
        )

    # ── Пакет ────────────────────────────────────────────
    @app.post("/v1/detect/batch", response_model=BatchDetectResponse)
    def detect_batch(req: BatchDetectRequest):
        if len(req.samples_b64) != len(req.predicted_labels):
            raise HTTPException(
                400,
                f"samples_b64 ({len(req.samples_b64)}) и "
                f"predicted_labels ({len(req.predicted_labels)}) "
                f"должны быть одной длины",
            )
        if len(req.samples_b64) == 0:
            raise HTTPException(400, "пустой пакет")

        combine = req.combine or default_combine
        threshold = req.threshold if req.threshold is not None else default_threshold
        local_detector = SentinelDetector(
            profile, z_threshold=threshold, combine=combine
        )

        results = []
        scores = []
        n_adv = 0
        for i, (b64, label) in enumerate(
                zip(req.samples_b64, req.predicted_labels)):
            try:
                raw = base64.b64decode(b64, validate=True)
            except Exception as e:
                raise HTTPException(400, f"invalid base64 at index {i}: {e}")
            v = local_detector.check(raw, label)
            results.append(DetectResponse(
                is_adversarial=v.is_adversarial,
                score=v.score,
                z_frag=v.z_frag, z_H=v.z_H, z_unique=v.z_unique,
                predicted_label=v.predicted_label, reason=v.reason,
            ))
            scores.append(v.score)
            n_adv += int(v.is_adversarial)

        import numpy as np
        return BatchDetectResponse(
            results=results,
            n_adversarial=n_adv,
            n_total=len(results),
            mean_score=float(np.mean(scores)),
        )

    return app