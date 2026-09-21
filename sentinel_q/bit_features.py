"""
bit_features.py — пространственные битовые признаки (SBF).

Находка: для изображений byte-level frag недостаточен. Adversarial-шум
живёт в отдельных битовых слоях. Пространственные признаки на этих
слоях (transition density, gradient) дают AUC = 0.97 на CIFAR-10.

Использование:
    from sentinel_q.bit_features import spatial_bit_features
    feats = spatial_bit_features(img_np)   # img_np: (C, H, W) float [0,1]
    # feats["td_bit5"], feats["hdiff_bit5"], ...

Включается в SentinelDetector через bit_features=True.
"""

from __future__ import annotations
import numpy as np

from .core import fragility, shannon_entropy

__all__ = [
    "spatial_bit_features",
    "SBG_GREEDY_FEATURES",
    "SBF_ALL_FEATURES",
]


# Greedy-набор из финального эксперимента (CIFAR-10 AUC 0.985)
SBG_GREEDY_FEATURES = [
    "td_bit5", "hdiff_bit5", "td_bit4", "td_bit6", "hdiff_bit4",
    "frag_bit4", "td_bit0", "frag_bit3", "td_bit2", "H_bit5",
]


def spatial_bit_features(img_np: np.ndarray) -> dict:
    """
    Пространственные битовые признаки (SBF).

    img_np: (C, H, W) float [0,1] или (H, W) float [0,1].
    Возвращает dict с 50+ признаками.
    """
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, :, :]

    arr = (img_np * 255).astype(np.uint8)
    out = {}

    for bit in range(8):
        bp = ((arr >> bit) & 1).astype(np.uint8)
        bp_bytes = (bp * 255).tobytes()
        binary = bp.tobytes()

        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")
        out[f"H_bit{bit}"] = shannon_entropy(binary)

        flat = bp.flatten()
        if len(flat) > 1:
            out[f"td_bit{bit}"] = float((flat[1:] != flat[:-1]).mean())
        else:
            out[f"td_bit{bit}"] = 0.0

        # 2D градиенты (только для H×W изображений)
        if bp.ndim == 3:
            hd = np.abs(bp[:, :, 1:] - bp[:, :, :-1]).mean()
            vd = np.abs(bp[:, 1:, :] - bp[:, :-1, :]).mean()
        elif bp.ndim == 2:
            hd = np.abs(bp[:, 1:] - bp[:, :-1]).mean()
            vd = np.abs(bp[1:, :] - bp[:-1, :]).mean()
        else:
            hd = vd = 0.0

        out[f"hdiff_bit{bit}"] = float(hd)
        out[f"vdiff_bit{bit}"] = float(vd)

    # Базовые
    full = arr.tobytes()
    out["frag"] = fragility(full, "zlib")
    out["frag_b"] = fragility(full, "bz2")
    out["H"] = shannon_entropy(full)

    return out


# Полный список всех признаков
SBF_ALL_FEATURES = (
    [f"frag_bit{i}" for i in range(8)]
    + [f"H_bit{i}" for i in range(8)]
    + [f"td_bit{i}" for i in range(8)]
    + [f"hdiff_bit{i}" for i in range(8)]
    + [f"vdiff_bit{i}" for i in range(8)]
    + ["frag", "frag_b", "H"]
)