"""
cifar_balanced.py — сбалансированный train, отдельная калибровка.

Проблема: train=300+30 (ratio 10:1) недостаточен. AUC падает 0.97→0.86.
Причина: 30 adversarial примеров не покрывают разнообразие атак.

Решение:
  - Train: 300 benign + 300 adv (1:1)
  - Calib: 100 benign (для порога FPR)
  - Test:  100 benign + 100 adv

Плюс:
  - Сравнение 6 классификаторов
  - Ансамбль (soft voting)
  - Финальный экспорт лучшей модели

Запуск:
    python cifar_balanced.py --n_benign 500
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


def all_features(img_np):
    """Полный набор bit-признаков."""
    out = {}
    arr = (img_np * 255).astype(np.uint8)

    full = arr.tobytes()
    out["frag"] = fragility(full, "zlib")
    out["frag_b"] = fragility(full, "bz2")
    out["H"] = shannon_entropy(full)
    out["nu"] = len(np.unique(np.frombuffer(full, dtype=np.uint8)))

    for bit in range(8):
        bp = ((arr >> bit) & 1).astype(np.uint8)
        bp_bytes = (bp * 255).tobytes()
        binary = bp.tobytes()

        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")
        out[f"H_bit{bit}"] = shannon_entropy(binary)

        flat = bp.flatten()
        out[f"td_bit{bit}"] = float(
            (flat[1:] != flat[:-1]).mean() if len(flat) > 1 else 0.0
        )
        out[f"hdiff_bit{bit}"] = float(
            np.abs(bp[:, 1:] - bp[:, :-1]).mean()
        )
        out[f"vdiff_bit{bit}"] = float(
            np.abs(bp[1:, :] - bp[:-1, :]).mean()
        )

    return out


def extract_all(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return all_features(img)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(one, [arr[i] for i in range(len(arr))]))
    return results


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


def get_feature_keys():
    """Все bit-признаки."""
    return (
        ["frag", "frag_b", "H", "nu"]
        + [f"frag_bit{i}" for i in range(8)]
        + [f"H_bit{i}" for i in range(8)]
        + [f"td_bit{i}" for i in range(8)]
        + [f"hdiff_bit{i}" for i in range(8)]
        + [f"vdiff_bit{i}" for i in range(8)]
    )


def run(n_benign=500, n_workers=4, target_fpr=0.05):
    from sklearn.metrics import roc_auc_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   ExtraTreesClassifier,
                                   HistGradientBoostingClassifier)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import VotingClassifier

    print("=" * 80)
    print(f"Сбалансированный train + отдельная калибровка")
    print(f"Device: {DEVICE}, n_benign={n_benign}")
    print("=" * 80)

    model, x_all, y_all = load_all()
    total = len(x_all)

    # Split: 60% train / 20% calib / 20% test
    n_train = min(int(0.6 * n_benign), int(0.6 * total))
    n_calib = min(int(0.2 * n_benign), int(0.2 * total))
    n_test = min(n_benign - n_train - n_calib, total - n_train - n_calib)

    x_b_train = x_all[:n_train].to(DEVICE)
    y_b_train = y_all[:n_train].to(DEVICE)
    x_b_calib = x_all[n_train:n_train + n_calib].to(DEVICE)
    y_b_calib = y_all[n_train:n_train + n_calib].to(DEVICE)
    x_b_test = x_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)
    y_b_test = y_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)

    print(f"\n[0] Split: train={n_train}, calib={n_calib}, test={n_test}")

    # ── Генерация adversarial
    print(f"\n[1] PGD-20 (1:1 train, 1:1 calib, 1:1 test) ...")
    t0 = time.time()

    # Train adv: генерируется из тех же benign train
    x_a_train = pgd_attack(model, x_b_train.clone(), y_b_train)
    # Calib adv: из benign calib (для оценки, не для калибровки)
    x_a_calib = pgd_attack(model, x_b_calib.clone(), y_b_calib)
    # Test adv: из benign test
    x_a_test = pgd_attack(model, x_b_test.clone(), y_b_test)

    print(f"    {time.time()-t0:.1f}s")

    # Проверка ASR
    with torch.no_grad():
        asr_train = (model(x_b_train).argmax(1)
                      != model(x_a_train).argmax(1)).float().mean().item()
        asr_test = (model(x_b_test).argmax(1)
                     != model(x_a_test).argmax(1)).float().mean().item()
    print(f"    ASR train={asr_train:.3f}, test={asr_test:.3f}")

    # ── Признаки
    print(f"\n[2] Извлечение признаков ...")
    t0 = time.time()
    fb_train = extract_all(x_b_train, n_workers)
    fa_train = extract_all(x_a_train, n_workers)
    fb_calib = extract_all(x_b_calib, n_workers)
    fa_calib = extract_all(x_a_calib, n_workers)
    fb_test = extract_all(x_b_test, n_workers)
    fa_test = extract_all(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = get_feature_keys()
    print(f"    Признаков: {len(keys)}")

    def to_X(feature_list):
        return np.array([[f[k] for k in keys] for f in feature_list])

    X_train = np.vstack([to_X(fb_train), to_X(fa_train)])
    y_train = np.concatenate([np.zeros(len(fb_train)),
                               np.ones(len(fa_train))])

    X_calib_b = to_X(fb_calib)
    X_calib_a = to_X(fa_calib)

    X_test_b = to_X(fb_test)
    X_test_a = to_X(fa_test)
    y_test = np.concatenate([np.zeros(len(fb_test)),
                              np.ones(len(fa_test))])

    print(f"    Train: {len(fb_train)}+{len(fa_train)} = {len(X_train)}")
    print(f"    Calib: {len(fb_calib)}+{len(fa_calib)}")
    print(f"    Test:  {len(fb_test)}+{len(fa_test)}")

    # ── Классификаторы
    classifiers = {
        "LR": make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=5000),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.1,
            random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, max_depth=10,
            random_state=42, n_jobs=-1,
        ),
        "MLP": make_pipeline(
            RobustScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64),
                           max_iter=1000, random_state=42,
                           early_stopping=True,
                           validation_fraction=0.15),
        ),
    }

    print(f"\n[3] Обучение:")
    print(f"    {'classifier':<22} {'AUC':>8} {'FPR@5%':>8} "
          f"{'det@5%':>8} {'det@10%':>8} {'best F1':>8}")
    print("    " + "-" * 62)

    results = {}
    for name, clf in classifiers.items():
        t0 = time.time()
        clf.fit(X_train, y_train)

        prob_b_calib = clf.predict_proba(X_calib_b)[:, 1]
        prob_a_calib = clf.predict_proba(X_calib_a)[:, 1]
        prob_b_test = clf.predict_proba(X_test_b)[:, 1]
        prob_a_test = clf.predict_proba(X_test_a)[:, 1]

        y_scores = np.concatenate([prob_b_test, prob_a_test])
        auc = roc_auc_score(y_test, y_scores)

        # Порог под FPR=5% на calib
        sorted_calib = np.sort(prob_b_calib)[::-1]
        idx5 = min(int(0.05 * len(sorted_calib)), len(sorted_calib) - 1)
        thr5 = sorted_calib[idx5]
        fpr5 = float((prob_b_test > thr5).mean())
        det5 = float((prob_a_test > thr5).mean())

        # Порог под FPR=10%
        idx10 = min(int(0.10 * len(sorted_calib)), len(sorted_calib) - 1)
        thr10 = sorted_calib[idx10]
        fpr10 = float((prob_b_test > thr10).mean())
        det10 = float((prob_a_test > thr10).mean())

        # Best F1 на test
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 100):
            preds = (y_scores > thr).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        elapsed = time.time() - t0
        results[name] = {
            "auc": auc, "fpr5": fpr5, "det5": det5,
            "det10": det10, "best_f1": best_f1,
            "best_thr": best_thr, "time": elapsed,
            "probs": (prob_b_test, prob_a_test),
        }

        marker = ""
        if det5 >= 0.75:
            marker = " ✓✓✓"
        elif det5 >= 0.60:
            marker = " ✓"

        print(f"    {name:<22} {auc:>8.4f} {fpr5:>8.3f} "
              f"{det5:>8.3f} {det10:>8.3f} {best_f1:>8.3f}{marker}")

    # ── Ансамбль
    print(f"\n[4] Ансамбль (soft voting):")
    try:
        voters = [
            ("rf", classifiers["RandomForest"]),
            ("gb", classifiers["GradientBoosting"]),
            ("et", classifiers["ExtraTrees"]),
        ]
        ens = VotingClassifier(voters, voting="soft")
        ens.fit(X_train, y_train)

        prob_b_calib = ens.predict_proba(X_calib_b)[:, 1]
        prob_b_test = ens.predict_proba(X_test_b)[:, 1]
        prob_a_test = ens.predict_proba(X_test_a)[:, 1]

        y_scores = np.concatenate([prob_b_test, prob_a_test])
        auc = roc_auc_score(y_test, y_scores)

        sorted_calib = np.sort(prob_b_calib)[::-1]
        idx5 = min(int(0.05 * len(sorted_calib)), len(sorted_calib) - 1)
        thr5 = sorted_calib[idx5]
        fpr5 = float((prob_b_test > thr5).mean())
        det5 = float((prob_a_test > thr5).mean())

        idx10 = min(int(0.10 * len(sorted_calib)), len(sorted_calib) - 1)
        thr10 = sorted_calib[idx10]
        fpr10 = float((prob_b_test > thr10).mean())
        det10 = float((prob_a_test > thr10).mean())

        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 100):
            preds = (y_scores > thr).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        results["Ensemble"] = {
            "auc": auc, "fpr5": fpr5, "det5": det5,
            "det10": det10, "best_f1": best_f1,
            "best_thr": best_thr, "time": 0,
            "probs": (prob_b_test, prob_a_test),
        }
        marker = " ✓✓✓" if det5 >= 0.75 else (" ✓" if det5 >= 0.60 else "")
        print(f"    {'Ensemble':<22} {auc:>8.4f} {fpr5:>8.3f} "
              f"{det5:>8.3f} {det10:>8.3f} {best_f1:>8.3f}{marker}")
    except Exception as e:
        print(f"    Ensemble ошибка: {e}")

    # ── Итог
    print(f"\n{'=' * 80}")
    print("ИТОГ")
    print(f"{'=' * 80}")

    sorted_r = sorted(results.items(), key=lambda kv: -kv[1]["det5"])
    print(f"\n  Top-3 по det@FPR=5%:")
    for i, (name, r) in enumerate(sorted_r[:3]):
        print(f"    {i+1}. {name:<22} AUC={r['auc']:.4f}  "
              f"FPR={r['fpr5']:.3f}  det={r['det5']:.3f}")

    best_name, best = sorted_r[0]
    print(f"\n  Лучший: {best_name}")
    print(f"    AUC = {best['auc']:.4f}")
    print(f"    FPR@5%: {best['fpr5']:.3f}, det: {best['det5']:.3f}")
    print(f"    FPR@10%: {best['det10']:.3f} det")

    if best["det5"] >= 0.75:
        print(f"\n✓✓✓ ЦЕЛЬ ДОСТИГНУТА: det@FPR=5% ≥ 0.75")
    elif best["det5"] >= 0.60:
        print(f"\n✓ Хорошо: det@FPR=5% ≥ 0.60")
    else:
        print(f"\n~ Не достигнута: det@FPR=5% = {best['det5']:.3f}")

    # Экспорт
    save_data = {
        "classifier_name": best_name,
        "classifier": classifiers.get(best_name) if best_name != "Ensemble" else ens,
        "keys": keys,
        "auc": best["auc"], "fpr5": best["fpr5"], "det5": best["det5"],
        "det10": best["det10"], "best_f1": best["best_f1"],
    }
    with open("sentinel_q_cifar_final.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nСохранено: sentinel_q_cifar_final.pkl")

    # CSV
    with open("cifar_balanced_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "auc", "fpr5", "det5",
                    "det10", "best_f1"])
        for name, r in results.items():
            w.writerow([name, f"{r['auc']:.4f}",
                        f"{r['fpr5']:.4f}", f"{r['det5']:.4f}",
                        f"{r['det10']:.4f}", f"{r['best_f1']:.4f}"])
    print(f"Сохранено: cifar_balanced_results.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for name, r in results.items():
            prob_b, prob_a = r["probs"]
            fpr_arr, tpr_arr, _ = roc_curve(
                np.concatenate([np.zeros(len(prob_b)),
                                np.ones(len(prob_a))]),
                np.concatenate([prob_b, prob_a]),
            )
            ax.plot(fpr_arr, tpr_arr, linewidth=2,
                    label=f"{name} (AUC={r['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.axvline(0.05, color="orange", linestyle="--",
                   alpha=0.5, label="FPR=5%")
        ax.set_xlabel("FPR")
        ax.set_ylabel("Detection")
        ax.set_title("ROC — CIFAR-10 balanced")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        names = list(results.keys())
        dets = [results[n]["det5"] for n in names]
        aucs = [results[n]["auc"] for n in names]
        x = np.arange(len(names))
        w = 0.35
        ax.barh(x - w/2, aucs, w, color="steelblue", label="AUC")
        ax.barh(x + w/2, dets, w, color="crimson",
                label="det@FPR=5%")
        ax.set_yticks(x)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0.75, color="green", linestyle="--",
                   alpha=0.5, label="target det=0.75")
        ax.set_xlabel("score")
        ax.set_title("AUC vs det@FPR=5%")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig("cifar_balanced.png", dpi=120)
        print("Сохранено: cifar_balanced.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()

    run(n_benign=args.n_benign, n_workers=args.jobs,
        target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()