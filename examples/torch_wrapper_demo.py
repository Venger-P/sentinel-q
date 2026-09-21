"""
Демонстрация SentinelWrapper на MNIST.

Требует:
  - обученная MNIST-модель (adv_cache/mnist_cnn.pt из Kvorb)
  - профиль (profile_mnist.json)
  - данные (adv_cache/benign.npz, adv_cache/adversarial.npz)

Запуск:
    python examples/torch_wrapper_demo.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel_q.torch_wrapper import (
    SentinelWrapper,
    AdversarialDetectedError,
)


# ── Мини-модель (должна совпадать с обученной) ──────────────

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Пути ───────────────────────────────────────────────────

KVORB = ROOT.parent / "Kvorb" / "V5.0"
MODEL_PATH  = KVORB / "adv_cache" / "mnist_cnn.pt"
BENIGN_PATH = KVORB / "adv_cache" / "benign.npz"
ADV_PATH    = KVORB / "adv_cache" / "adversarial.npz"
PROFILE_PATH = ROOT / "profile_mnist.json"


def load_model(device):
    model = SmallCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def main():
    print("=" * 66)
    print("SentinelWrapper demo (MNIST)")
    print("=" * 66)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model = load_model(device)
    print(f"Модель загружена: {MODEL_PATH}")
    print(f"Профиль: {PROFILE_PATH}")

    b = np.load(BENIGN_PATH)
    a = np.load(ADV_PATH)
    n = 100
    xb = torch.from_numpy(b["x"][:n]).float().to(device)
    xa = torch.from_numpy(a["x"][:n]).float().to(device)

    # ── 1. Baseline без обёртки ─────────────────────────
    print("\n[1] Baseline (без обёртки):")
    with torch.no_grad():
        p_b = base_model(xb).argmax(1).cpu()
        p_a = base_model(xa).argmax(1).cpu()
    print(f"  Benign предсказаний: {p_b.tolist()[:10]}...")
    print(f"  Adversarial предсказаний: {p_a.tolist()[:10]}...")

    # ── 2. mode="log" ───────────────────────────────────
    print("\n[2] SentinelWrapper(mode='log'):")
    wrapped = SentinelWrapper(base_model, str(PROFILE_PATH), mode="log")
    wrapped.eval()

    with torch.no_grad():
        _ = wrapped(xb)
        _ = wrapped(xa)

    stats = wrapped.stats()
    print(f"  n_total:   {stats['n_total']}")
    print(f"  n_adv:     {stats['n_adv']}")
    print(f"  adv_rate:  {stats['adv_rate']*100:.1f}%")
    print(f"  mean_score:{stats['mean_score']:.3f}")
    print(f"  max_score: {stats['max_score']:.3f}")

    # ── 3. mode="raise" ─────────────────────────────────
    print("\n[3] SentinelWrapper(mode='raise'):")
    strict = SentinelWrapper(base_model, str(PROFILE_PATH), mode="raise")
    strict.eval()

    # Benign должен пройти
    try:
        with torch.no_grad():
            _ = strict(xb)
        print("  benign: прошёл без ошибки ✓")
    except AdversarialDetectedError as e:
        print(f"  benign: ложная тревога! {e}")

    # Adversarial должен быть заблокирован
    try:
        with torch.no_grad():
            _ = strict(xa)
        print("  adversarial: НЕ заблокирован (плохо)")
    except AdversarialDetectedError as e:
        print(f"  adversarial: заблокирован ✓")
        print(f"    количество: {len(e.indices)}")
        print(f"    max score:  {max(e.scores):.3f}")

    # ── 4. mode="zero" ──────────────────────────────────
    print("\n[4] SentinelWrapper(mode='zero'):")
    zeroer = SentinelWrapper(base_model, str(PROFILE_PATH), mode="zero")
    zeroer.eval()

    with torch.no_grad():
        logits_b = zeroer(xb)
        logits_a = zeroer(xa)

    n_zero_b = int((logits_b.abs().sum(dim=1) == 0).sum())
    n_zero_a = int((logits_a.abs().sum(dim=1) == 0).sum())
    print(f"  Обнулённых логитов (benign):      {n_zero_b}/{n}")
    print(f"  Обнулённых логитов (adversarial): {n_zero_a}/{n}")

    # ── 5. mode="warn" ──────────────────────────────────
    print("\n[5] SentinelWrapper(mode='warn'):")
    warner = SentinelWrapper(base_model, str(PROFILE_PATH), mode="warn")
    warner.eval()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            _ = warner(xa)
        print(f"  warnings: {len(caught)}")
        if caught:
            print(f"    первое: {str(caught[0].message)[:80]}...")

    # ── 6. Порог: как меняется trade-off ────────────────
    print("\n[6] Влияние порога на FP/FN:")
    print(f"  {'threshold':>10} {'FP (benign→adv)':>18} {'FN (adv→benign)':>18}")
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        w = SentinelWrapper(base_model, str(PROFILE_PATH),
                            mode="log", threshold=thr)
        w.eval()
        with torch.no_grad():
            _ = w(xb)
            _ = w(xa)
        s = w.stats_data.scores
        n_adv_b = sum(1 for i in range(n) if s[i] > thr)
        n_adv_a = sum(1 for i in range(n, 2*n) if s[i] > thr)
        fp = n_adv_b
        fn = n - n_adv_a
        print(f"  {thr:>10.1f} {fp:>18d} {fn:>18d}")

    print("\nГотово.")


if __name__ == "__main__":
    main()