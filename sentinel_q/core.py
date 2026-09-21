"""
Ядро Sentinel-Q: вычисление байтовых характеристик образца.

Функции:
    image_stats(data)      — frag, H, n_unique для bytes
    image_stats_from_array — то же для np.ndarray

Без зависимостей от ML-фреймворков.
"""

from __future__ import annotations
import zlib
import bz2
import lzma

import numpy as np

__all__ = [
    "shannon_entropy",
    "n_unique_bytes",
    "fragility",
    "byte_stats",
    "QorbDescriptor",
]


# ── Базовые операции ─────────────────────────────────────────

def shannon_entropy(data: bytes) -> float:
    if len(data) == 0:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    p = counts[counts > 0] / len(arr)
    return float(-np.sum(p * np.log2(p)))


def n_unique_bytes(data: bytes) -> int:
    if len(data) == 0:
        return 0
    return int(len(np.unique(np.frombuffer(data, dtype=np.uint8))))


def _compress(data: bytes, algo: str) -> bytes:
    if algo == "zlib":
        return zlib.compress(data, 9)
    if algo == "bz2":
        return bz2.compress(data, 9)
    if algo == "lzma":
        return lzma.compress(data, preset=9)
    raise ValueError(f"unknown compressor: {algo}")


def _add_noise(data: bytes, p: float, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.frombuffer(data, dtype=np.uint8).copy()
    mask = rng.random(len(arr)) < p
    arr[mask] = rng.integers(0, 256, size=int(mask.sum()), dtype=np.uint8)
    return arr.tobytes()


def fragility(data: bytes,
              algo: str = "zlib",
              p_levels=(0.01, 0.02, 0.05, 0.10, 0.20)) -> float:
    """
    frag = (R(0) - mean_p R(p)) / R(0)
    где R(p) = |X| / |C(X_p)|.
    """
    if len(data) == 0:
        return 0.0
    R0 = len(data) / max(len(_compress(data, algo)), 1)
    if R0 <= 1.0:
        return 0.0
    Rs = []
    for p in p_levels:
        noisy = _add_noise(data, p, seed=int(p * 1e6))
        Rs.append(len(data) / max(len(_compress(noisy, algo)), 1))
    return float((R0 - float(np.mean(Rs))) / R0)


def byte_stats(data: bytes) -> dict:
    """Полный набор байтовых характеристик."""
    return {
        "frag":     float(fragility(data, "zlib")),
        "frag_z":   float(fragility(data, "zlib")),
        "frag_b":   float(fragility(data, "bz2")),
        "H":        shannon_entropy(data),
        "n_unique": n_unique_bytes(data),
        "size":     len(data),
    }


# ── QorbDescriptor (полная версия с сигмоидой) ──────────────

from scipy.optimize import curve_fit


def _sigmoid(p, R0, p_star, w):
    return R0 / (1.0 + np.exp((p - p_star) / max(w, 1e-9)))


class QorbDescriptor:
    """Полная метрика w и CV через фит сигмоиды."""

    def __init__(self, compressors=("zlib", "bz2"),
                 n_points=15, n_trials=2, max_p=0.4):
        self.compressors = compressors
        self.n_points = n_points
        self.n_trials = n_trials
        self.max_p = max_p

    def fit(self, data: bytes):
        self.n_bytes = len(data)
        self.H = shannon_entropy(data)
        ps = np.linspace(0.001, self.max_p, self.n_points)
        ws = []
        per_comp = {}
        import hashlib
        for name in self.compressors:
            R0 = len(data) / max(len(_compress(data, name)), 1)
            rs = []
            for p in ps:
                trials = []
                for t in range(self.n_trials):
                    seed = int(hashlib.md5(
                        f"{name}_{p:.6f}_{t}".encode()
                    ).hexdigest()[:8], 16) & 0xFFFFFFFF
                    noisy = _add_noise(data, p, seed)
                    trials.append(len(data) /
                                  max(len(_compress(noisy, name)), 1))
                rs.append(float(np.median(trials)))
            try:
                popt, _ = curve_fit(
                    _sigmoid, ps, np.array(rs),
                    p0=[R0, 0.05, 0.1],
                    bounds=([0.5, -0.01, 0.001],
                            [R0 * 2 + 1, 0.5, 10.0]),
                    maxfev=5000,
                )
                w = max(float(popt[2]), 0.001)
            except Exception:
                w = 10.0
            ws.append(w)
            per_comp[name] = {"w": w}
        self.w_per = per_comp
        self.w = float(np.median(ws))
        self.CV = (float(np.std(ws)) / self.w) if self.w > 0.001 else 0.0
        return self

    def report(self) -> str:
        return (f"QorbDescriptor: w={self.w:.4f}, CV={self.CV:.2%}, "
                f"H={self.H:.4f}, n={self.n_bytes}")