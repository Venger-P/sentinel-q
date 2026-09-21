"""
cifar_asymmetric.py — асимметричное обучение для контроля FPR (v2).

Исправление: multiprocessing → threading. На Windows spawn-режим
создаёт новый процесс с torch+CUDA в каждом воркере → OOM.
Threading решает проблему: zlib освобождает GIL.

Запуск:
    python cifar_asymmetric.py --n_benign 500 --ratio 10
"""

import argparse
import csv
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy
from sentinel_q.bit_features import SBG_GREEDY_FEATURES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Модель ──────────────────────────────────────────────────

class SmallCNN32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ── Bit-признаки ────────────────────────────────────────────

# Биты, которые реально нужны (из SBG_GREEDY_FEATURES)
_REQUIRED_BITS = {0, 2, 3, 4, 5, 6}


def _one_image_features(img_np):
    """Признаки одного изображения (C, H, W) float [0, 1]."""
    out = {}
    arr = (img_np * 255).astype(np.uint8)

    # Считаем только нужные биты
    for bit in _REQUIRED_BITS:
        bp = ((arr >> bit) & 1).astype(np.uint8)

        # frag на байтовом представлении слоя
        if f"frag_bit{bit}" in SBG_GREEDY_FEATURES:
            bp_bytes = (bp * 255).tobytes()
            out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")

        # H на бинарном слое
        if f"H_bit{bit}" in SBG_GREEDY_FEATURES:
            binary = bp.tobytes()
            out[f"H_bit{bit}"] = shannon_entropy(binary)

        # Transition density
        if f"td_bit{bit}" in SBG_GREEDY_FEATURES:
            flat = bp.flatten()
            out[f"td_bit{bit}"] = float(
                (flat[1:] != flat[:-1]).mean() if len(flat) > 1 else 0.0
            )

        # Horizontal gradient
        if f"hdiff_bit{bit}" in SBG_GREEDY_FEATURES:
            out[f"hdiff_bit{bit}"] = float(
                np.abs(bp[:, 1:] - bp[:, :-1]).mean()
            )

    # Возвращаем в порядке SBG_GREEDY_FEATURES
    return [out.get(k, 0.0) for k in SBG_GREEDY_FEATURES]


def extract_features(x_tensor, n_workers=4):
    """Извлечение признаков через ThreadPoolExecutor."""
    arr = x_tensor.cpu().numpy()
    # Освобождаем CUDA перед CPU-работой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_one_image_features, [arr[i] for i in range(len(arr))]))
    return np.array(results)


def pgd_attack(model, x, y, eps=0.05, alpha=None, n_iter=20):
    if alpha is None:
        alpha = eps / 4
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = (x + delta).clamp(0, 1).detach()
    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = (x_adv - x).clamp(-eps, eps)
            x_adv = (x + delta).clamp(0, 1)
    return x_adv.detach()


def load_all():
    cache = Path("../Kvorb/V5.0/adv_cache_cifar")
    if not cache.exists():
        cache = Path("adv_cache_cifar")
    d = np.load(cache / "benign.npz")
    x = torch.from_numpy(d["x"]).float()
    y = torch.from_numpy(d["y"]).long()
    model_path = cache / "cifar_cnn.pt"
    if not model_path.exists():
        model_path = cache / "standard.pt"
    model = SmallCNN32().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, x, y


# ── Основной эксперимент ────────────────────────────────────

def run(n_benign=500, ratio=10, n_workers=4, target_fpr=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    print("=" * 78)
    print(f"Асимметричное обучение: ratio benign:adv = {ratio}:1")
    print(f"Device: {DEVICE}, workers = {n_workers}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    total_available = len(x_all)
    print(f"\n[0] Доступно benign: {total_available}")

    # Split: 80% train, 20% test (но не больше, чем есть)
    n_train = min(int(0.8 * n_benign), int(0.8 * total_available))
    n_test = min(n_benign - n_train, total_available - n_train)
    if n_test < 10:
        n_train = max(20, total_available - 50)
        n_test = total_available - n_train

    print(f"    Используем: train={n_train}, test={n_test}")

    x_b_train = x_all[:n_train].to(DEVICE)
    y_b_train = y_all[:n_train].to(DEVICE)
    x_b_test = x_all[n_train:n_train + n_test].to(DEVICE)
    y_b_test = y_all[n_train:n_train + n_test].to(DEVICE)

    # Adv-примеры генерируем из того же benign, но с PGD
    n_adv_train = max(1, n_train // ratio)
    n_adv_test = max(1, n_test // ratio)

    x_a_train = x_all[:n_adv_train].to(DEVICE)
    y_a_train = y_all[:n_adv_train].to(DEVICE)
    x_a_test = x_all[n_train:n_train + n_adv_test].to(DEVICE)
    y_a_test = y_all[n_train:n_train + n_adv_test].to(DEVICE)

    print(f"\n[1] Train: {n_train} benign + {n_adv_train} adv")
    print(f"    Test:  {n_test} benign + {n_adv_test} adv")

    # PGD
    print(f"\n[2] Генерация PGD-20 ...")
    t0 = time.time()
    x_a_train = pgd_attack(model, x_a_train, y_a_train)
    x_a_test = pgd_attack(model, x_a_test, y_a_test)
    print(f"    {time.time()-t0:.1f}s")

    # ── Признаки (threading)
    print(f"\n[3] Извлечение bit-признаков ({n_workers} threads) ...")

    t0 = time.time()
    X_b_train = extract_features(x_b_train, n_workers=n_workers)
    print(f"    benign train ({len(X_b_train)}): {time.time()-t0:.1f}s")

    t0 = time.time()
    X_a_train = extract_features(x_a_train, n_workers=n_workers)
    print(f"    adv train ({len(X_a_train)}):    {time.time()-t0:.1f}s")

    t0 = time.time()
    X_b_test = extract_features(x_b_test, n_workers=n_workers)
    print(f"    benign test ({len(X_b_test)}):  {time.time()-t0:.1f}s")

    t0 = time.time()
    X_a_test = extract_features(x_a_test, n_workers=n_workers)
    print(f"    adv test ({len(X_a_test)}):     {time.time()-t0:.1f}s")

    # ── Обучение
    print(f"\n[4] Обучение LR (без class_weight) ...")
    X_train = np.vstack([X_b_train, X_a_train])
    y_train = np.concatenate([np.zeros(len(X_b_train)),
                               np.ones(len(X_a_train))])

    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000),
    )
    pipe.fit(X_train, y_train)

    # ── Оценка
    print(f"\n[5] Оценка на test:")
    prob_b = pipe.predict_proba(X_b_test)[:, 1]
    prob_a = pipe.predict_proba(X_a_test)[:, 1]

    auc = roc_auc_score(
        np.concatenate([np.zeros(len(prob_b)), np.ones(len(prob_a))]),
        np.concatenate([prob_b, prob_a]),
    )
    print(f"    AUC = {auc:.4f}")

    print(f"\n    {'threshold':>10} {'FPR':>8} {'Detection':>10} "
          f"{'F1-like':>10}")
    print("    " + "-" * 42)

    best = None
    for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        fpr = float((prob_b > thr).mean())
        det = float((prob_a > thr).mean())
        f1 = 2 * det * (1 - fpr) / max(det + (1 - fpr), 1e-9)
        if best is None or f1 > best["f1"]:
            best = {"thr": thr, "fpr": fpr, "det": det, "f1": f1}
        print(f"    {thr:>10.2f} {fpr:>8.3f} {det:>10.3f} {f1:>10.3f}")

    # ── Калибровка
    print(f"\n[6] Калибровка под target FPR = {target_fpr}:")
    candidates_sorted = np.sort(prob_b)[::-1]
    idx = min(int(target_fpr * len(candidates_sorted)),
              len(candidates_sorted) - 1)
    thr_calib = candidates_sorted[idx]
    fpr_calib = float((prob_b > thr_calib).mean())
    det_calib = float((prob_a > thr_calib).mean())
    print(f"    threshold = {thr_calib:.4f}")
    print(f"    FPR = {fpr_calib:.3f}")
    print(f"    Detection = {det_calib:.3f}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    print(f"\n  Ratio: {ratio}:1")
    print(f"  AUC: {auc:.4f}")
    print(f"  Best F1: FPR={best['fpr']:.3f}, det={best['det']:.3f}")
    print(f"  Target FPR: {fpr_calib:.3f}, det={det_calib:.3f}")

    if fpr_calib <= target_fpr + 0.02 and det_calib >= 0.75:
        print(f"\n✓✓ ЦЕЛЬ ДОСТИГНУТА: FPR ≤ {target_fpr*100:.0f}%, det ≥ 75%")
    elif fpr_calib <= 0.10 and det_calib >= 0.70:
        print(f"\n✓ Хорошо: FPR ≤ 10%, det ≥ 70%")
    else:
        print(f"\n~ Требует настройки")

    # Сохранение
    save_data = {
        "type": "asymmetric_lr",
        "pipeline": pipe,
        "threshold": float(thr_calib),
        "ratio": ratio,
        "keys": SBG_GREEDY_FEATURES,
        "auc": auc, "fpr": fpr_calib, "det": det_calib,
    }
    with open("sentinel_q_cifar_asymmetric.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nСохранено: sentinel_q_cifar_asymmetric.pkl")

    with open("cifar_asymmetric_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "fpr", "detection", "f1_like"])
        for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            fpr = float((prob_b > thr).mean())
            det = float((prob_a > thr).mean())
            f1 = 2 * det * (1 - fpr) / max(det + (1 - fpr), 1e-9)
            w.writerow([f"{thr:.2f}", f"{fpr:.4f}",
                        f"{det:.4f}", f"{f1:.4f}"])
    print(f"Сохранено: cifar_asymmetric_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500,
                    help="Всего benign (не больше, чем доступно)")
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()

    run(n_benign=args.n_benign, ratio=args.ratio,
        n_workers=args.jobs, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()