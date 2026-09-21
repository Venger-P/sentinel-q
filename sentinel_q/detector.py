"""
SentinelDetector — детекция adversarial-примеров по эталонному профилю.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .core import byte_stats
from .profile import ReferenceProfile


@dataclass
class Verdict:
    is_adversarial: bool
    score: float
    z_frag: float
    z_H: float
    z_unique: float
    predicted_label: int
    reason: str


class SentinelDetector:
    """
    Model-free детектор adversarial-примеров.

    Логика: образец считается adversarial, если его frag аномально
    низкий (и H аномально высокий) относительно эталона класса.

    combine:
        "weighted"  — 0.7 * frag_z + 0.3 * H_z   (по умолчанию)
        "frag_only" — только frag_z
        "mean"      — среднее по трём признакам
        "max"       — максимум из трёх
    """

    WEIGHTS = {
        "weighted":  {"frag": 0.7, "H": 0.3, "n_unique": 0.0},
        "frag_only": {"frag": 1.0, "H": 0.0, "n_unique": 0.0},
        "mean":      {"frag": 1/3, "H": 1/3, "n_unique": 1/3},
        "max":       {"frag": 1.0, "H": 1.0, "n_unique": 1.0},
    }

    def __init__(self, profile: ReferenceProfile,
                 z_threshold: float = 1.5,
                 combine: str = "weighted"):
        self.profile = profile
        self.z_threshold = z_threshold
        if combine not in self.WEIGHTS:
            raise ValueError(f"unknown combine: {combine}; "
                             f"choose from {list(self.WEIGHTS)}")
        self.combine = combine
        self.weights = self.WEIGHTS[combine]

    def _z_scores(self, sample: bytes, label: int) -> dict:
        s = self.profile.stats.get(label)
        if s is None:
            return {"frag": 0.0, "H": 0.0, "n_unique": 0.0}
        stats = byte_stats(sample)
        out = {}
        for f in ("frag", "H", "n_unique"):
            mu = s[f"{f}_mean"]
            sigma = max(s[f"{f}_std"], 1e-9)
            z = (stats[f] - mu) / sigma
            if f == "frag":
                z = -z  # ниже — подозрительнее
            out[f] = float(z)
        return out

    def _combine(self, z: dict) -> float:
        if self.combine == "max":
            return float(max(z.values()))
        return float(
            self.weights["frag"]     * z["frag"] +
            self.weights["H"]        * z["H"] +
            self.weights["n_unique"] * z["n_unique"]
        )

    def check(self, sample: bytes, predicted_label: int) -> Verdict:
        z = self._z_scores(sample, predicted_label)
        score = self._combine(z)
        is_adv = score > self.z_threshold
        if is_adv:
            reason = (f"frag z={z['frag']:+.2f}, "
                      f"H z={z['H']:+.2f}, "
                      f"n_unique z={z['n_unique']:+.2f}")
        else:
            reason = "consistent with benign profile"
        return Verdict(
            is_adversarial=is_adv,
            score=float(score),
            z_frag=z["frag"],
            z_H=z["H"],
            z_unique=z["n_unique"],
            predicted_label=predicted_label,
            reason=reason,
        )

    def check_batch(self, samples: list, labels: list) -> list:
        return [self.check(s, int(y)) for s, y in zip(samples, labels)]

    def batch_roc(self, samples: list, labels: list,
                  is_adversarial: list) -> dict:
        from sklearn.metrics import roc_auc_score, accuracy_score
        verdicts = self.check_batch(samples, labels)
        scores = [v.score for v in verdicts]
        preds = [int(v.is_adversarial) for v in verdicts]
        return {
            "roc_auc":  float(roc_auc_score(is_adversarial, scores)),
            "accuracy": float(accuracy_score(is_adversarial, preds)),
        }