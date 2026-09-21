"""
ReferenceProfile — эталонные распределения байтовых признаков.

Использование:
    profile = ReferenceProfile(label_space=[0..9])
    profile.build(benign_data, labels)
    profile.save("profile.json")
    profile = ReferenceProfile.load("profile.json")
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .core import byte_stats


class ReferenceProfile:
    """
    Эталонные распределения (frag, H, n_unique) по классам.

    Для каждого класса хранится:
      - mean, std, quantiles[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
      - n_samples
    """

    VERSION = "1.0"

    def __init__(self, label_space: Optional[Iterable] = None):
        self.label_space = list(label_space) if label_space else []
        self.stats: dict = {}
        self.n_samples_total = 0
        self.feature_names = ["frag", "H", "n_unique"]

    # ── Построение ─────────────────────────────────────────

    def build(self, samples: list, labels: list,
              verbose: bool = False):
        """
        samples: список bytes (сырые байтовые представления)
        labels:  список меток, соответствующих samples
        """
        if len(samples) != len(labels):
            raise ValueError("samples и labels должны быть одной длины")

        buckets: dict = {}
        for i, (s, y) in enumerate(zip(samples, labels)):
            y = int(y)
            if y not in buckets:
                buckets[y] = []
            buckets[y].append(byte_stats(s))
            if verbose and (i + 1) % 500 == 0:
                print(f"  обработано: {i+1}/{len(samples)}")

        for label, feats in buckets.items():
            arr = {k: np.array([f[k] for f in feats])
                   for k in self.feature_names}

            class_stats = {"n_samples": len(feats)}
            for k in self.feature_names:
                vals = arr[k]
                class_stats[f"{k}_mean"] = float(vals.mean())
                class_stats[f"{k}_std"] = float(vals.std())
                class_stats[f"{k}_q01"] = float(np.quantile(vals, 0.01))
                class_stats[f"{k}_q05"] = float(np.quantile(vals, 0.05))
                class_stats[f"{k}_q10"] = float(np.quantile(vals, 0.10))
                class_stats[f"{k}_q50"] = float(np.quantile(vals, 0.50))
                class_stats[f"{k}_q90"] = float(np.quantile(vals, 0.90))
                class_stats[f"{k}_q95"] = float(np.quantile(vals, 0.95))
                class_stats[f"{k}_q99"] = float(np.quantile(vals, 0.99))

            self.stats[label] = class_stats

        self.n_samples_total = len(samples)
        if not self.label_space:
            self.label_space = sorted(self.stats.keys())
        return self

    # ── Предсказание ───────────────────────────────────────

    def z_score(self, sample: bytes, label: int,
                feature: str = "frag") -> float:
        """Z-score образца относительно класса label по признаку feature."""
        if label not in self.stats:
            return 0.0
        s = self.stats[label]
        mu = s[f"{feature}_mean"]
        sigma = s[f"{feature}_std"]
        if sigma < 1e-9:
            return 0.0
        val = byte_stats(sample)[feature]
        return float((val - mu) / sigma)

    def quantile_rank(self, sample: bytes, label: int,
                      feature: str = "frag") -> float:
        """Позиция образца в распределении класса (0..1)."""
        if label not in self.stats:
            return 0.5
        val = byte_stats(sample)[feature]
        s = self.stats[label]
        qs = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
        vs = [s[f"{feature}_q{int(q * 100):02d}"] for q in qs]
        if val <= vs[0]:
            return 0.0
        if val >= vs[-1]:
            return 1.0
        for i in range(len(vs) - 1):
            if vs[i] <= val <= vs[i + 1]:
                t = (val - vs[i]) / max(vs[i + 1] - vs[i], 1e-12)
                return float(qs[i] + t * (qs[i + 1] - qs[i]))
        return 0.5

    # ── Сохранение ─────────────────────────────────────────

    def save(self, path):
        path = Path(path)
        payload = {
            "version": self.VERSION,
            "label_space": self.label_space,
            "n_samples_total": self.n_samples_total,
            "feature_names": self.feature_names,
            "stats": self.stats,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        p = cls(label_space=payload["label_space"])
        p.stats = {int(k): v for k, v in payload["stats"].items()}
        p.n_samples_total = payload["n_samples_total"]
        p.feature_names = payload["feature_names"]
        return p

    def summary(self) -> str:
        lines = [f"ReferenceProfile: {self.n_samples_total} samples, "
                 f"{len(self.stats)} classes"]
        for label in sorted(self.stats):
            s = self.stats[label]
            lines.append(
                f"  class {label:>3}: n={s['n_samples']:>5}  "
                f"frag={s['frag_mean']:.4f}±{s['frag_std']:.4f}  "
                f"H={s['H_mean']:.3f}±{s['H_std']:.3f}"
            )
        return "\n".join(lines)