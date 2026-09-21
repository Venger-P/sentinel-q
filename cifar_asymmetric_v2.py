"""
cifar_asymmetric_v2.py — асимметричное обучение с полным test set.

Исправление v1: test adversarial генерировался только из n_test/ratio
примеров (10 из 100). Это давало статистический шум.
Теперь: test adversarial — ПОЛНЫЙ n_test, а не подмножество.

Train остаётся асимметричным (ratio:1), test — сбалансированным.

Запуск:
    python cifar_asymmetric_v2.py --n_benign 500 --ratio 10
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


_REQUIRED_BITS = {0, 2, 3, 4, 5, 6}


def _one_image_features(img_np):
    out = {}
    arr = (img_np * 255).astype(np.uint8)
    for bit in _REQUIRED_BITS:
        bp = ((arr >> bit) & 1).astype(np.uint8)
        if f"frag_bit{bit}" in SBG_GREEDY_FEATURES:
            out[f"frag_bit{bit}"] = fragility((bp * 255).tobytes(), "zlib")
        if f"H_bit{bit}" in SBG_GREEDY_FEATURES:
            out[f"H_bit{bit}"] = shannon_entropy(bp.tobytes())
        if f"td_bit{bit}" in SBG_GREEDY_FEATURES:
            flat = bp.flatten()
            out[f"td_bit{bit}"] = float(
                (flat[1:] != flat[:-1]).mean() if len(flat) > 1 else 0.0
            )
        if f"hdiff_bit{bit}" in SBG_GREEDY_FEATURES:
            out[f"hdiff_bit{bit}"] = float(
                np.abs(bp[:, 1:] - bp[:, :-1]).mean()
            )
    return [out.get(k, 0.0) for k in SBG_GREEDY_FEATURES]


def extract_features(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_one_image_features,
                                 [arr[i] for i in range(len(arr))]))
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


def run(n_benign=500, ratio=10, n_workers=4, target_fpr=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    print("=" * 78)
    print(f"Асимметричное обучение v2 — ratio={ratio}:1")
    print(f"Device: {DEVICE}, workers = {n_workers}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    total = len(x_all)

    # Split: 60% train / 20% calib / 20% test
    n_train = min(int(0.6 * n_benign), int(0.6 * total))
    n_calib = min(int(0.2 * n_benign), int(0.2 * total))
    n_test = min(n_benign - n_train - n_calib,
                 total - n_train - n_calib)
    if n_test < 20:
        n_test = total - n_train - n_calib

    print(f"\n[0] Split: train={n_train}, calib={n_calib}, test={n_test}")

    # Benign splits
    x_b_train = x_all[:n_train].to(DEVICE)
    y_b_train = y_all[:n_train].to(DEVICE)

    x_b_calib = x_all[n_train:n_train + n_calib].to(DEVICE)
    y_b_calib = y_all[n_train:n_train + n_calib].to(DEVICE)

    x_b_test = x_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)
    y_b_test = y_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)

    # Adversarial: обучаем на подмножестве (ratio:1), тестируем на всех
    n_adv_train = max(1, n_train // ratio)

    x_a_train = x_b_train[:n_adv_train].clone()
    y_a_train = y_b_train[:n_adv_train]

    print(f"\n[1] Train: {n_train} benign + {n_adv_train} adv "
          f"(ratio={ratio})")
    print(f"    Calib: {n_calib} benign + {n_calib} adv (full)")
    print(f"    Test:  {n_test} benign + {n_test} adv (full)")

    # PGD: train (subset), calib (all), test (all)
    print(f"\n[2] Генерация PGD-20 ...")
    t0 = time.time()
    x_a_train = pgd_attack(model, x_a_train, y_a_train)
    x_a_calib = pgd_attack(model, x_b_calib, y_b_calib)  # все calib
    x_a_test = pgd_attack(model, x_b_test, y_b_test)     # все test
    print(f"    {time.time()-t0:.1f}s")

    # ── Признаки
    print(f"\n[3] Извлечение bit-признаков ...")

    t0 = time.time()
    X_b_train = extract_features(x_b_train, n_workers)
    X_a_train = extract_features(x_a_train, n_workers)
    print(f"    train: {time.time()-t0:.1f}s")

    t0 = time.time()
    X_b_calib = extract_features(x_b_calib, n_workers)
    X_a_calib = extract_features(x_a_calib, n_workers)
    print(f"    calib: {time.time()-t0:.1f}s")

    t0 = time.time()
    X_b_test = extract_features(x_b_test, n_workers)
    X_a_test = extract_features(x_a_test, n_workers)
    print(f"    test:  {time.time()-t0:.1f}s")

    # ── Обучение
    print(f"\n[4] Обучение LR (no class_weight) ...")
    X_train = np.vstack([X_b_train, X_a_train])
    y_train = np.concatenate([np.zeros(len(X_b_train)),
                               np.ones(len(X_a_train))])
    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000),
    )
    pipe.fit(X_train, y_train)

    # ── Оценка на test
    print(f"\n[5] Оценка на test ({n_test} + {n_test}):")
    prob_b_test = pipe.predict_proba(X_b_test)[:, 1]
    prob_a_test = pipe.predict_proba(X_a_test)[:, 1]
    prob_b_calib = pipe.predict_proba(X_b_calib)[:, 1]
    prob_a_calib = pipe.predict_proba(X_a_calib)[:, 1]

    y_true = np.concatenate([np.zeros(len(prob_b_test)),
                              np.ones(len(prob_a_test))])
    y_scores = np.concatenate([prob_b_test, prob_a_test])
    auc = roc_auc_score(y_true, y_scores)
    print(f"    AUC = {auc:.4f}")

    # ── Threshold sweep
    print(f"\n[6] Threshold sweep на test:")
    print(f"    {'thr':>6} {'FPR':>8} {'det':>8} {'prec':>8} "
          f"{'recall':>8} {'F1':>8}")
    print("    " + "-" * 54)

    sweep = []
    for thr in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9]:
        preds = (y_scores > thr).astype(int)
        fpr = float((prob_b_test > thr).mean())
        det = float((prob_a_test > thr).mean())
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        sweep.append({"thr": thr, "fpr": fpr, "det": det,
                       "prec": prec, "rec": rec, "f1": f1})
        print(f"    {thr:>6.2f} {fpr:>8.3f} {det:>8.3f} "
              f"{prec:>8.3f} {rec:>8.3f} {f1:>8.3f}")

    # ── Калибровка на отдельной выборке
    print(f"\n[7] Калибровка (target FPR = {target_fpr}):")
    sorted_calib = np.sort(prob_b_calib)[::-1]
    idx = min(int(target_fpr * len(sorted_calib)), len(sorted_calib) - 1)
    thr_calib = sorted_calib[idx]

    fpr_calib = float((prob_b_calib > thr_calib).mean())
    det_calib = float((prob_a_calib > thr_calib).mean())

    fpr_test = float((prob_b_test > thr_calib).mean())
    det_test = float((prob_a_test > thr_calib).mean())

    print(f"    threshold = {thr_calib:.4f}")
    print(f"    Calib:  FPR={fpr_calib:.3f}, det={det_calib:.3f}")
    print(f"    Test:   FPR={fpr_test:.3f}, det={det_test:.3f}")

    # ── Выбор лучшего
    print(f"\n[8] Выбор лучшего threshold по F1-like:")
    best = max(sweep,
                key=lambda r: 2 * r["det"] * (1 - r["fpr"]) /
                max(r["det"] + (1 - r["fpr"]), 1e-9))
    print(f"    thr = {best['thr']:.2f}, FPR = {best['fpr']:.3f}, "
          f"det = {best['det']:.3f}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    print(f"\n  AUC: {auc:.4f}")
    print(f"\n  {'config':<28} {'FPR':>8} {'det':>8} {'F1':>8}")
    print("  " + "-" * 54)
    print(f"  {'Best F1 (test)':<28} {best['fpr']:>8.3f} "
          f"{best['det']:>8.3f} "
          f"{best['f1']:>8.3f}")
    print(f"  {'Calibrated (target=0.05)':<28} {fpr_test:>8.3f} "
          f"{det_test:>8.3f} —")

    if fpr_test <= 0.05 and det_test >= 0.75:
        print(f"\n✓✓✓ ЦЕЛЬ ДОСТИГНУТА: FPR ≤ 5%, det ≥ 75%")
    elif fpr_test <= 0.10 and det_test >= 0.70:
        print(f"\n✓✓ ХОРОШО: FPR ≤ 10%, det ≥ 70%")
    elif best["fpr"] <= 0.10 and best["det"] >= 0.80:
        print(f"\n✓ Best F1: FPR={best['fpr']:.3f}, det={best['det']:.3f}")
    else:
        print(f"\n~ Требует настройки")

    # Сохранение
    save_data = {
        "type": "asymmetric_lr_v2",
        "pipeline": pipe,
        "threshold_calibrated": float(thr_calib),
        "threshold_best_f1": float(best["thr"]),
        "ratio": ratio,
        "keys": SBG_GREEDY_FEATURES,
        "auc": auc,
        "fpr_calib": fpr_test,
        "det_calib": det_test,
    }
    with open("sentinel_q_cifar_asymmetric_v2.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nСохранено: sentinel_q_cifar_asymmetric_v2.pkl")

    with open("cifar_asymmetric_v2_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "fpr", "det", "prec", "rec", "f1"])
        for r in sweep:
            w.writerow([f"{r['thr']:.2f}", f"{r['fpr']:.4f}",
                        f"{r['det']:.4f}", f"{r['prec']:.4f}",
                        f"{r['rec']:.4f}", f"{r['f1']:.4f}"])
    print(f"Сохранено: cifar_asymmetric_v2_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--ratio", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()

    run(n_benign=args.n_benign, ratio=args.ratio,
        n_workers=args.jobs, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()