"""
End-to-end проверка Sentinel-Q HTTP-сервиса на MNIST.

Требует работающего сервера:
    sentinel-q serve --profile profile_mnist.json

Запуск:
    python examples/http_demo.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel_q.client import SentinelClient


def to_bytes(x, n=None):
    """x: (N,1,28,28) float[0,1] -> список bytes."""
    if n is not None:
        x = x[:n]
    arr = (x * 255).astype(np.uint8)
    return [arr[i, 0].tobytes() for i in range(len(arr))]


def main():
    base = "http://127.0.0.1:8000"
    c = SentinelClient(base)

    print("=" * 60)
    print("Sentinel-Q HTTP demo")
    print("=" * 60)

    # Health check
    try:
        h = c.healthz()
        print(f"\nHealth: {h}")
    except Exception as e:
        print(f"\nСервер не отвечает: {e}")
        print("Запустите в другом терминале:")
        print("  sentinel-q serve --profile profile_mnist.json")
        sys.exit(1)

    # Загрузка данных
    benign_path = ROOT.parent / "Kvorb" / "V5.0" / "adv_cache" / "benign.npz"
    adv_path    = ROOT.parent / "Kvorb" / "V5.0" / "adv_cache" / "adversarial.npz"

    if not benign_path.exists():
        print(f"\nФайл не найден: {benign_path}")
        sys.exit(1)

    b = np.load(benign_path)
    a = np.load(adv_path)

    n = 100
    xb, yb = b["x"][:n], b["y"][:n]
    xa, ya = a["x"][:n], a["y"][:n]

    print(f"\nДанные: {n} benign + {n} adversarial")

    # Benign
    print("\n[1] Отправка benign ...")
    rb = c.detect_batch(to_bytes(xb), yb.tolist())
    pct_b = rb["n_adversarial"] / rb["n_total"] * 100
    print(f"  Подозрительных: {rb['n_adversarial']}/{rb['n_total']} "
          f"({pct_b:.1f}%)")
    print(f"  Средний score:  {rb['mean_score']:.3f}")

    # Adversarial
    print("\n[2] Отправка adversarial ...")
    ra = c.detect_batch(to_bytes(xa), ya.tolist())
    pct_a = ra["n_adversarial"] / ra["n_total"] * 100
    print(f"  Подозрительных: {ra['n_adversarial']}/{ra['n_total']} "
          f"({pct_a:.1f}%)")
    print(f"  Средний score:  {ra['mean_score']:.3f}")

    # Метрики
    print("\n[3] Метрики через HTTP:")
    from sklearn.metrics import roc_auc_score, accuracy_score

    y_true = [0] * n + [1] * n
    scores = ([r["score"] for r in rb["results"]] +
              [r["score"] for r in ra["results"]])
    preds = ([int(r["is_adversarial"]) for r in rb["results"]] +
             [int(r["is_adversarial"]) for r in ra["results"]])

    print(f"  ROC-AUC:  {roc_auc_score(y_true, scores):.4f}")
    print(f"  Accuracy: {accuracy_score(y_true, preds):.4f}")

    # Примеры вердиктов
    print("\n[4] Примеры вердиктов (adversarial):")
    for i in range(min(5, len(ra["results"]))):
        r = ra["results"][i]
        mark = "ADV" if r["is_adversarial"] else "ok "
        print(f"  [{mark}] #{i} score={r['score']:+.3f}  {r['reason']}")

    print("\nГотово.")


if __name__ == "__main__":
    main()