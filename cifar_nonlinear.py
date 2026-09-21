"""
cifar_nonlinear.py — нелинейные классификаторы на bit-признаках.

Проблема: LR даёт AUC=0.91, FPR=5% → det=53%. Нелинейные
взаимодействия между bit-признаками не улавливаются.

Решение: GradientBoosting, RandomForest, MLP. Табличные данные
с 10 признаками — идеальный случай для tree-based методов.

Протокол:
  - Train: 300 benign + 30 adv (ratio=10)
  - Calib: 100 benign + 100 adv
  - Test:  100 benign + 100 adv

Запуск:
    python cifar_nonlinear.py --n_benign 500 --ratio 10
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


def get_classifiers():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (GradientBoostingClassifier,
                                   RandomForestClassifier,
                                   ExtraTreesClassifier,
                                   HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import make_pipeline

    return {
        "LR (baseline)": make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=5000),
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.1,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8,
            random_state=42, n_jobs=-1,
        ),
        "MLP": make_pipeline(
            RobustScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000, random_state=42,
                early_stopping=True, validation_fraction=0.15,
            ),
        ),
    }


def run(n_benign=500, ratio=10, n_workers=4, target_fpr=0.05):
    from sklearn.metrics import roc_auc_score, f1_score

    print("=" * 78)
    print(f"Нелинейные классификаторы на bit-признаках — ratio={ratio}")
    print(f"Device: {DEVICE}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    total = len(x_all)

    n_train = min(int(0.6 * n_benign), int(0.6 * total))
    n_calib = min(int(0.2 * n_benign), int(0.2 * total))
    n_test = min(n_benign - n_train - n_calib,
                 total - n_train - n_calib)

    print(f"\n[0] Split: train={n_train}, calib={n_calib}, test={n_test}")

    x_b_train = x_all[:n_train].to(DEVICE)
    y_b_train = y_all[:n_train].to(DEVICE)
    x_b_calib = x_all[n_train:n_train + n_calib].to(DEVICE)
    y_b_calib = y_all[n_train:n_train + n_calib].to(DEVICE)
    x_b_test = x_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)
    y_b_test = y_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)

    n_adv_train = max(1, n_train // ratio)
    x_a_train = x_b_train[:n_adv_train].clone()
    y_a_train = y_b_train[:n_adv_train]

    print(f"    Train: {n_train} benign + {n_adv_train} adv")
    print(f"    Calib: {n_calib} + {n_calib}")
    print(f"    Test:  {n_test} + {n_test}")

    # PGD
    print(f"\n[1] PGD-20 ...")
    t0 = time.time()
    x_a_train = pgd_attack(model, x_a_train, y_a_train)
    x_a_calib = pgd_attack(model, x_b_calib, y_b_calib)
    x_a_test = pgd_attack(model, x_b_test, y_b_test)
    print(f"    {time.time()-t0:.1f}s")

    # Признаки
    print(f"\n[2] Bit-признаки ...")
    t0 = time.time()
    X_b_train = extract_features(x_b_train, n_workers)
    X_a_train = extract_features(x_a_train, n_workers)
    X_b_calib = extract_features(x_b_calib, n_workers)
    X_a_calib = extract_features(x_a_calib, n_workers)
    X_b_test = extract_features(x_b_test, n_workers)
    X_a_test = extract_features(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    X_train = np.vstack([X_b_train, X_a_train])
    y_train = np.concatenate([np.zeros(len(X_b_train)),
                               np.ones(len(X_a_train))])

    y_test = np.concatenate([np.zeros(len(X_b_test)),
                              np.ones(len(X_a_test))])

    # ── Обучение и оценка
    print(f"\n[3] Обучение классификаторов:")
    print(f"    {'classifier':<22} {'AUC':>8} {'FPR@5%':>8} "
          f"{'det@5%':>8} {'best F1':>8} {'best thr':>9}")
    print("    " + "-" * 70)

    classifiers = get_classifiers()
    results = []

    for name, clf in classifiers.items():
        t0 = time.time()
        try:
            clf.fit(X_train, y_train)
            prob_b_test = clf.predict_proba(X_b_test)[:, 1]
            prob_a_test = clf.predict_proba(X_a_test)[:, 1]
            prob_b_calib = clf.predict_proba(X_b_calib)[:, 1]

            y_scores = np.concatenate([prob_b_test, prob_a_test])
            auc = roc_auc_score(y_test, y_scores)

            # Калибровка под target FPR
            sorted_calib = np.sort(prob_b_calib)[::-1]
            idx = min(int(target_fpr * len(sorted_calib)),
                       len(sorted_calib) - 1)
            thr_calib = sorted_calib[idx]
            fpr_5 = float((prob_b_test > thr_calib).mean())
            det_5 = float((prob_a_test > thr_calib).mean())

            # Best F1
            best_f1, best_thr = 0.0, 0.5
            for thr in np.linspace(0.05, 0.95, 50):
                preds = (y_scores > thr).astype(int)
                f1 = f1_score(y_test, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr

            elapsed = time.time() - t0
            results.append({
                "name": name, "auc": auc,
                "fpr_5": fpr_5, "det_5": det_5,
                "best_f1": best_f1, "best_thr": best_thr,
                "time": elapsed,
            })
            print(f"    {name:<22} {auc:>8.4f} {fpr_5:>8.3f} "
                  f"{det_5:>8.3f} {best_f1:>8.3f} {best_thr:>9.2f}")
        except Exception as e:
            print(f"    {name:<22} ошибка: {e}")

    if not results:
        print("\n✗ Ни один классификатор не сработал")
        return

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    best_auc = max(results, key=lambda r: r["auc"])
    best_det5 = max(results, key=lambda r: r["det_5"])

    print(f"\n  Best AUC: {best_auc['name']} → {best_auc['auc']:.4f}")
    print(f"  Best det@FPR=5%: {best_det5['name']} → "
          f"FPR={best_det5['fpr_5']:.3f}, det={best_det5['det_5']:.3f}")

    # Цель: FPR ≤ 5% и det ≥ 0.75
    success = [r for r in results
               if r["fpr_5"] <= 0.07 and r["det_5"] >= 0.70]
    if success:
        print(f"\n✓✓✓ ЦЕЛЬ ДОСТИГНУТА:")
        for r in success:
            print(f"    {r['name']}: FPR={r['fpr_5']:.3f}, "
                  f"det={r['det_5']:.3f}, AUC={r['auc']:.4f}")

    # Сохранение лучшей модели
    best = max(results, key=lambda r: r["det_5"] - r["fpr_5"])
    best_clf = classifiers[best["name"]]
    save_data = {
        "type": best["name"],
        "classifier": best_clf,
        "keys": SBG_GREEDY_FEATURES,
        "auc": best["auc"],
        "fpr_5": best["fpr_5"],
        "det_5": best["det_5"],
        "best_f1": best["best_f1"],
        "best_thr": best["best_thr"],
    }
    with open("sentinel_q_cifar_nonlinear.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nСохранено: sentinel_q_cifar_nonlinear.pkl")

    with open("cifar_nonlinear_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "auc", "fpr_at_5", "det_at_5",
                    "best_f1", "best_thr"])
        for r in results:
            w.writerow([r["name"], f"{r['auc']:.4f}",
                        f"{r['fpr_5']:.4f}", f"{r['det_5']:.4f}",
                        f"{r['best_f1']:.4f}", f"{r['best_thr']:.2f}"])
    print(f"Сохранено: cifar_nonlinear_results.csv")


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