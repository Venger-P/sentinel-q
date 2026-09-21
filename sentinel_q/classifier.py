"""
AttackClassifier — определение типа атаки по байтовым характеристикам.

Обучается на размеченных данных (benign, FGSM, PGD, CW, ...).
"""

from __future__ import annotations
from pathlib import Path
import pickle

import numpy as np

from .core import byte_stats


class AttackClassifier:

    def __init__(self):
        self._pipeline = None
        self.labels_ = None

    @staticmethod
    def _features(samples: list) -> np.ndarray:
        rows = []
        for s in samples:
            st = byte_stats(s)
            rows.append([
                st["frag"],
                st["H"],
                st["n_unique"],
                st["frag"] / max(st["H"], 1e-9),
                np.log1p(st["size"]),
            ])
        return np.array(rows, dtype=float)

    def fit(self, samples: list, labels: list):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        X = self._features(samples)
        y = np.array(labels)
        self._pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        self._pipeline.fit(X, y)
        self.labels_ = sorted(set(int(v) for v in y))
        return self

    def predict(self, sample: bytes) -> int:
        X = self._features([sample])
        return int(self._pipeline.predict(X)[0])

    def predict_proba(self, sample: bytes) -> dict:
        X = self._features([sample])
        proba = self._pipeline.predict_proba(X)[0]
        return {int(c): float(p)
                for c, p in zip(self._pipeline.classes_, proba)}

    def score(self, samples: list, labels: list) -> float:
        from sklearn.metrics import accuracy_score
        X = self._features(samples)
        return float(accuracy_score(labels, self._pipeline.predict(X)))

    def save(self, path: str | Path):
        Path(path).write_bytes(pickle.dumps({
            "pipeline": self._pipeline,
            "labels": self.labels_,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "AttackClassifier":
        payload = pickle.loads(Path(path).read_bytes())
        obj = cls()
        obj._pipeline = payload["pipeline"]
        obj.labels_ = payload["labels"]
        return obj