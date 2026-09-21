"""
End-to-end демо Sentinel-Q на MNIST.

Требует наличия:
    adv_cache/benign.npz        — чистые примеры (x, y)
    adv_cache/adversarial.npz   — FGSM-примеры (x, y)

Данные ищутся в ../Kvorb/V5.0/adv_cache/ (пути автоматические).

Запуск:
    python examples/demo_mnist.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel_q import ReferenceProfile, SentinelDetector


# ── Поиск данных ────────────────────────────────────────────

def find_data():
    """Ищет adv_cache в нескольких местах."""
    candidates = [
        ROOT / "adv_cache",
        ROOT.parent / "Kvorb" / "V5.0" / "adv_cache",
        ROOT.parent / "adv_cache",
    ]
    for c in candidates:
        if (c / "benign.npz").exists() and (c / "adversarial.npz").exists():
            return c
    raise FileNotFoundError(
        "Не найдена папка adv_cache ни в одном из мест:\n" +
        "\n".join(f"  {c}" for c in candidates)
    )


def to_bytes(x):
    arr = (x * 255).astype(np.uint8)
    return [arr[i, 0].tobytes() for i in range(len(arr))]


def main():
    print("=" * 70)
    print("Sentinel-Q — демо на MNIST")
    print("=" * 70)

    cache = find_data()
    print(f"Данные: {cache}")

    benign = np.load(cache / "benign.npz")
    adv = np.load(cache / "adversarial.npz")

    x_b, y_b = benign["x"], benign["y"]
    x_a, y_a = adv["x"], adv["y"]

    n = min(500, len(x_b), len(x_a))
    x_b, y_b = x_b[:n], y_b[:n]
    x_a, y_a = x_a[:n], y_a[:n]

    print(f"\nПримеров: {n} benign + {n} adversarial")

    # 1. Строим профиль на benign
    print("\n[1] Построение эталонного профиля (benign) ...")
    profile = ReferenceProfile().build(to_bytes(x_b), y_b.tolist())
    print(profile.summary())

    # 2. Детектируем adversarial
    print("\n[2] Детекция ...")
    detector = SentinelDetector(profile, z_threshold=1.5, combine="weighted")

    samples = to_bytes(x_b) + to_bytes(x_a)
    labels = list(y_b) + list(y_a)
    is_adv = [False] * n + [True] * n

    verdicts = detector.check_batch(samples, labels)
    scores = np.array([v.score for v in verdicts])
    preds = np.array([int(v.is_adversarial) for v in verdicts])
    truth = np.array(is_adv)

    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    tn = int(((preds == 0) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    acc = (tp + tn) / len(truth)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(truth, scores)
        print(f"  ROC-AUC:   {auc:.4f}")
    except ImportError:
        print("  ROC-AUC:   (sklearn не установлен)")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # 3. Зависимость от порога
    print("\n[3] Зависимость от порога:")
    print(f"  {'threshold':>10} {'accuracy':>10} "
          f"{'precision':>10} {'recall':>10}")
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        d = SentinelDetector(profile, z_threshold=thr, combine="weighted")
        v = d.check_batch(samples, labels)
        p = np.array([int(x.is_adversarial) for x in v])
        acc_t = (p == truth).mean()
        tp_t = ((p == 1) & (truth == 1)).sum()
        fp_t = ((p == 1) & (truth == 0)).sum()
        fn_t = ((p == 0) & (truth == 1)).sum()
        prec = tp_t / max(tp_t + fp_t, 1)
        rec = tp_t / max(tp_t + fn_t, 1)
        print(f"  {thr:>10.1f} {acc_t:>10.4f} "
              f"{prec:>10.4f} {rec:>10.4f}")

    print("\nГотово.")


if __name__ == "__main__":
    main()