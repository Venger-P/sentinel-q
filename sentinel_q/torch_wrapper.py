"""
SentinelWrapper — прозрачная обёртка над PyTorch-моделью.

Автоматически проверяет вход на adversarial-примеры перед forward.
Не меняет API модели: работает как nn.Module.

Режимы:
    "log"    — логирует в self.stats, пропускает дальше (мониторинг)
    "warn"   — то же, но с warnings.warn
    "raise"  — бросает AdversarialDetectedError (firewall)
    "zero"   — возвращает нулевые логиты (модель "отказывается" отвечать)

Использование:
    from sentinel_q.torch_wrapper import SentinelWrapper

    model = SentinelWrapper(
        pretrained_model,
        profile="profile_mnist.json",
        mode="log",
    )
    y = model(x)              # проверка автоматически
    print(model.stats())      # {'n_total': 100, 'n_adv': 12, ...}

Требует:
    pip install torch
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .profile import ReferenceProfile
from .detector import SentinelDetector


# ── Исключения ──────────────────────────────────────────────

class AdversarialDetectedError(RuntimeError):
    """Бросается, когда SentinelWrapper обнаружил adversarial-вход."""

    def __init__(self, message: str, indices: list, scores: list):
        super().__init__(message)
        self.indices = indices
        self.scores = scores


# ── Статистика ──────────────────────────────────────────────

@dataclass
class WrapperStats:
    n_total: int = 0
    n_adv: int = 0
    n_batches: int = 0
    scores_sum: float = 0.0
    scores: list = field(default_factory=list)

    def update(self, verdicts):
        self.n_batches += 1
        for v in verdicts:
            self.n_total += 1
            self.scores_sum += v.score
            self.scores.append(v.score)
            if v.is_adversarial:
                self.n_adv += 1

    def to_dict(self) -> dict:
        mean = self.scores_sum / self.n_total if self.n_total > 0 else 0.0
        return {
            "n_total": self.n_total,
            "n_adv": self.n_adv,
            "adv_rate": self.n_adv / self.n_total if self.n_total > 0 else 0.0,
            "n_batches": self.n_batches,
            "mean_score": mean,
            "max_score": float(np.max(self.scores)) if self.scores else 0.0,
            "min_score": float(np.min(self.scores)) if self.scores else 0.0,
        }

    def reset(self):
        self.n_total = 0
        self.n_adv = 0
        self.n_batches = 0
        self.scores_sum = 0.0
        self.scores.clear()


# ── Вспомогательные ─────────────────────────────────────────

def _tensor_batch_to_bytes(x: torch.Tensor) -> list:
    """
    Преобразует батч изображений в список bytes.

    Поддерживает (N, C, H, W). Для C=1 использует только канал 0
    (получается 2D представление, как в эксперименте).
    """
    if x.dim() != 4:
        raise ValueError(f"ожидался тензор (N, C, H, W), получен {tuple(x.shape)}")

    arr = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    out = []
    for i in range(len(arr)):
        if arr.shape[1] == 1:
            out.append(arr[i, 0].tobytes())
        else:
            out.append(arr[i].tobytes())
    return out


# ── Обёртка ─────────────────────────────────────────────────

class SentinelWrapper(nn.Module):
    """
    Обёртка над PyTorch-моделью с автоматической проверкой входа.

    Параметры
    ---------
    model : nn.Module
        Любая модель: классификатор, feature extractor и т.д.
    profile : str | ReferenceProfile
        Путь к JSON-профилю или загруженный профиль.
    mode : str
        "log" | "warn" | "raise" | "zero"
    combine : str
        Способ комбинирования z-оценок ("weighted" по умолчанию).
    threshold : float
        Порог score (по умолчанию 1.5).
    check_enabled : bool
        Если False — обёртка просто пропускает forward без проверки.
    use_predicted_label : bool
        Если True (по умолчанию), метка класса берётся из argmax логитов.
        Если False, метка должна быть передана в forward как labels.
    """

    def __init__(
        self,
        model: nn.Module,
        profile,
        mode: str = "log",
        combine: str = "weighted",
        threshold: float = 1.5,
        check_enabled: bool = True,
        use_predicted_label: bool = True,
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.check_enabled = check_enabled
        self.use_predicted_label = use_predicted_label

        if isinstance(profile, str):
            profile = ReferenceProfile.load(profile)
        self.profile = profile
        self.detector = SentinelDetector(
            profile, z_threshold=threshold, combine=combine
        )
        self.stats_data = WrapperStats()

        if mode not in ("log", "warn", "raise", "zero"):
            raise ValueError(
                f"unknown mode: {mode}; "
                f"choose from log, warn, raise, zero"
            )

    # ── Управление ────────────────────────────────────────

    def enable_check(self):
        self.check_enabled = True

    def disable_check(self):
        self.check_enabled = False

    def reset_stats(self):
        self.stats_data.reset()

    def stats(self) -> dict:
        return self.stats_data.to_dict()

    # ── Проверка ──────────────────────────────────────────

    def _run_check(self, x: torch.Tensor, labels: torch.Tensor):
        samples = _tensor_batch_to_bytes(x)
        label_list = labels.detach().cpu().tolist()
        verdicts = self.detector.check_batch(samples, label_list)
        self.stats_data.update(verdicts)
        return verdicts

    def _handle_verdicts(self, verdicts, logits):
        adv_idx = [i for i, v in enumerate(verdicts) if v.is_adversarial]
        if not adv_idx:
            return logits

        scores = [verdicts[i].score for i in adv_idx]

        if self.mode == "log":
            # Ничего не делаем — статистика уже обновлена
            pass
        elif self.mode == "warn":
            import warnings
            warnings.warn(
                f"Sentinel-Q: обнаружено {len(adv_idx)} adversarial-примеров "
                f"в батче. Indices: {adv_idx[:10]}..."
                + (" (первые 10)" if len(adv_idx) > 10 else ""),
                RuntimeWarning,
            )
        elif self.mode == "raise":
            raise AdversarialDetectedError(
                f"Sentinel-Q: обнаружено {len(adv_idx)} adversarial-примеров "
                f"в батче размера {len(verdicts)}. "
                f"Max score = {max(scores):.3f}",
                indices=adv_idx,
                scores=scores,
            )
        elif self.mode == "zero":
            # Обнуляем логиты для атакованных примеров
            logits = logits.clone()
            for i in adv_idx:
                logits[i] = 0.0

        return logits

    # ── Forward ───────────────────────────────────────────

    def forward(self, x, labels: Optional[torch.Tensor] = None):
        """
        Параметры
        ---------
        x : torch.Tensor, (N, C, H, W)
            Входные данные.
        labels : torch.Tensor, (N,), optional
            Истинные метки. Если None и use_predicted_label=True,
            берутся из argmax forward-выхода модели.
            Если use_predicted_label=False — labels обязателен.

        Возвращает
        ----------
        logits : torch.Tensor
            Выход модели (возможно, модифицированный в режиме "zero").
        """
        logits = self.model(x)

        if not self.check_enabled:
            return logits

        if labels is None:
            if not self.use_predicted_label:
                raise ValueError(
                    "labels обязателен, если use_predicted_label=False"
                )
            with torch.no_grad():
                labels = logits.argmax(dim=1)

        verdicts = self._run_check(x, labels)
        logits = self._handle_verdicts(verdicts, logits)
        return logits

    # ── Делегирование атрибутов модели ───────────────────

    def __getattr__(self, name):
        """Проксирует неизвестные атрибуты к внутренней модели."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def train(self, mode: bool = True):
        """Переключает train/eval режим для вложенной модели."""
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)