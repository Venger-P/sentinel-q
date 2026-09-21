"""
cifar_calibrate.py — калибровка порога и снижение FPR.

Проблема: FPR = 17.3% при threshold=0.5. Для firewall это много.
Решение: калибровать порог и применить post-hoc методы.

Методы:
  1. Threshold tuning — выбрать порог под целевую FPR
  2. Platt scaling — калибровка вероятностей
  3. Isotonic regression — нелинейная калибровка
  4. Ensemble — усреднение по seed
  5. Профильный подход — вместо LR использовать z-score по классам

Цель: FPR ≤ 5% при detection ≥ 0.75.

Запуск:
    python cifar_calibrate.py --n 1000
    python cifar_calibrate.py --n 1000 --target_fpr 0.05
"""

import argparse
import csv
import pickle
import sys
import time
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

def real_bit_features(img_np):
    out = {}
    arr = (img_np * 255).astype(np.uint8)
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
    return {k: out[k] for k in SBG_GREEDY_FEATURES}


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


# ── Калибровка ──────────────────────────────────────────────

def calibrate_threshold(prob_b, prob_a, target_fpr=0.05):
    """
    Находит threshold, при котором FPR ≤ target_fpr.
    Возвращает (threshold, fpr_actual, detection).
    """
    all_probs = np.concatenate([prob_b, prob_a])
    # Кандидаты — квантили prob_b
    candidates = np.quantile(prob_b, np.linspace(0.5, 0.999, 200))
    best = None
    for thr in candidates:
        fpr = (prob_b > thr).mean()
        if fpr <= target_fpr:
            det = (prob_a > thr).mean()
            return float(thr), float(fpr), float(det)
    # Fallback — максимальный thr
    thr = prob_b.max() + 1e-6
    return float(thr), 0.0, 0.0


def platt_scaling(prob_raw, y_true):
    """Калибровка вероятностей через логистическую регрессию."""
    from sklearn.linear_model import LogisticRegression
    X = np.log(prob_raw / (1 - prob_raw + 1e-9)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(X, y_true)
    return lr


def isotonic_calibration(prob_raw, y_true):
    """Непараметрическая калибровка."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_raw, y_true)
    return iso


# ── Основной эксперимент ────────────────────────────────────

def run(n_samples=1000, target_fpr=0.05):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    print("=" * 78)
    print(f"Калибровка порога для Sentinel-Q v2 (CIFAR-10)")
    print(f"Device: {DEVICE}, target FPR = {target_fpr}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    n = min(n_samples, len(x_all))
    x_b = x_all[:n].to(DEVICE)
    y = y_all[:n].to(DEVICE)

    # Train/test split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(0.5 * n)      # 50% train
    n_calib = int(0.25 * n)     # 25% calibration
    n_test = n - n_train - n_calib

    train_idx = perm[:n_train]
    calib_idx = perm[n_train:n_train + n_calib]
    test_idx = perm[n_train + n_calib:]

    x_train = x_b[train_idx]; y_train = y[train_idx]
    x_calib = x_b[calib_idx]; y_calib = y[calib_idx]
    x_test = x_b[test_idx];   y_test = y[test_idx]

    print(f"\nSplit: train={n_train}, calib={n_calib}, test={n_test}")

    # Генерация PGD
    print(f"\n[1] Генерация PGD-20 ...")
    x_train_adv = pgd_attack(model, x_train, y_train)
    x_calib_adv = pgd_attack(model, x_calib, y_calib)
    x_test_adv = pgd_attack(model, x_test, y_test)

    with torch.no_grad():
        pred_train = model(x_train).argmax(1)
        asr_train = (pred_train != model(x_train_adv).argmax(1)).float().mean().item()
        pred_calib = model(x_calib).argmax(1)
        asr_calib = (pred_calib != model(x_calib_adv).argmax(1)).float().mean().item()
        pred_test = model(x_test).argmax(1)
        asr_test = (pred_test != model(x_test_adv).argmax(1)).float().mean().item()
    print(f"    ASR: train={asr_train:.3f}, calib={asr_calib:.3f}, "
          f"test={asr_test:.3f}")

    # ── Извлечение признаков
    print(f"\n[2] Извлечение bit-признаков ...")
    t0 = time.time()

    def extract(x_tensor):
        arr = x_tensor.cpu().numpy()
        return np.array([[real_bit_features(arr[i])[k]
                            for k in SBG_GREEDY_FEATURES]
                           for i in range(len(arr))])

    X_train_b = extract(x_train)
    X_train_a = extract(x_train_adv)
    X_calib_b = extract(x_calib)
    X_calib_a = extract(x_calib_adv)
    X_test_b = extract(x_test)
    X_test_a = extract(x_test_adv)
    print(f"    {time.time()-t0:.1f}s")

    # ── Обучение LR на train
    print(f"\n[3] Обучение LR на train ...")
    X_tr = np.vstack([X_train_b, X_train_a])
    y_tr = np.concatenate([np.zeros(len(X_train_b)),
                            np.ones(len(X_train_a))])
    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    pipe.fit(X_tr, y_tr)

    # ── Baseline (threshold = 0.5)
    print(f"\n[4] Baseline (threshold = 0.5):")
    prob_calib_b = pipe.predict_proba(X_calib_b)[:, 1]
    prob_calib_a = pipe.predict_proba(X_calib_a)[:, 1]
    prob_test_b = pipe.predict_proba(X_test_b)[:, 1]
    prob_test_a = pipe.predict_proba(X_test_a)[:, 1]

    fpr_05 = float((prob_test_b > 0.5).mean())
    det_05 = float((prob_test_a > 0.5).mean())
    auc_test = roc_auc_score(
        np.concatenate([np.zeros(len(prob_test_b)), np.ones(len(prob_test_a))]),
        np.concatenate([prob_test_b, prob_test_a]),
    )
    print(f"    FPR = {fpr_05:.3f}, Detection = {det_05:.3f}, "
          f"AUC = {auc_test:.4f}")

    # ── Threshold tuning на calibration
    print(f"\n[5] Threshold tuning на calibration (target FPR={target_fpr}):")
    thr, fpr_c, det_c = calibrate_threshold(prob_calib_b, prob_calib_a,
                                              target_fpr=target_fpr)
    print(f"    Выбран threshold = {thr:.4f}")
    print(f"    FPR на calib = {fpr_c:.3f}")
    print(f"    Detection на calib = {det_c:.3f}")

    # Проверка на test
    fpr_test_tuned = float((prob_test_b > thr).mean())
    det_test_tuned = float((prob_test_a > thr).mean())
    print(f"    На test: FPR = {fpr_test_tuned:.3f}, "
          f"Detection = {det_test_tuned:.3f}")

    # ── Плато: threshold vs FPR/detection
    print(f"\n[6] Кривая threshold → FPR/detection:")
    print(f"    {'threshold':>10} {'FPR':>8} {'detect':>9} {'F1-like':>9}")
    print("    " + "-" * 40)
    curve = []
    for thr_i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        fpr_i = float((prob_test_b > thr_i).mean())
        det_i = float((prob_test_a > thr_i).mean())
        f1_like = 2 * det_i * (1 - fpr_i) / max(det_i + (1 - fpr_i), 1e-9)
        curve.append({"thr": thr_i, "fpr": fpr_i, "det": det_i,
                       "f1": f1_like})
        print(f"    {thr_i:>10.2f} {fpr_i:>8.3f} {det_i:>9.3f} "
              f"{f1_like:>9.3f}")

    # ── Platt scaling на calibration
    print(f"\n[7] Platt scaling:")
    prob_calib_all = np.concatenate([prob_calib_b, prob_calib_a])
    y_calib_all = np.concatenate([np.zeros(len(prob_calib_b)),
                                    np.ones(len(prob_calib_a))])
    platt = platt_scaling(prob_calib_all, y_calib_all)
    prob_test_b_platt = platt.predict_proba(
        np.log(prob_test_b / (1 - prob_test_b + 1e-9)).reshape(-1, 1)
    )[:, 1]
    prob_test_a_platt = platt.predict_proba(
        np.log(prob_test_a / (1 - prob_test_a + 1e-9)).reshape(-1, 1)
    )[:, 1]
    fpr_platt = float((prob_test_b_platt > 0.5).mean())
    det_platt = float((prob_test_a_platt > 0.5).mean())
    print(f"    FPR = {fpr_platt:.3f}, Detection = {det_platt:.3f}")

    # ── Итоговая таблица
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")
    print(f"\n  {'Метод':<26} {'FPR':>8} {'Detection':>10} {'AUC':>8}")
    print("  " + "-" * 56)

    methods = [
        ("Baseline (thr=0.5)", fpr_05, det_05, auc_test),
        (f"Tuned (target={target_fpr})", fpr_test_tuned, det_test_tuned, auc_test),
        ("Platt scaling", fpr_platt, det_platt, auc_test),
    ]
    for name, fpr, det, auc in methods:
        print(f"  {name:<26} {fpr:>8.3f} {det:>10.3f} {auc:>8.4f}")

    # Выбор лучшего
    best = max(methods, key=lambda m: 2 * m[2] * (1 - m[1]) /
                max(m[2] + (1 - m[1]), 1e-9))
    print(f"\n  Лучший метод: {best[0]}")
    print(f"    FPR = {best[1]:.3f}, Detection = {best[2]:.3f}")

    if best[1] <= 0.05 and best[2] >= 0.75:
        print(f"\n✓✓ ЦЕЛЬ ДОСТИГНУТА")
        print(f"    FPR ≤ 5% и Detection ≥ 75%")
    elif best[1] <= 0.10:
        print(f"\n✓ ХОРОШО: FPR ≤ 10%")
    else:
        print(f"\n~ FPR всё ещё высок")

    # Сохранение
    with open("cifar_calibration.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "fpr", "detection", "f1_like"])
        for c in curve:
            w.writerow([f"{c['thr']:.2f}", f"{c['fpr']:.4f}",
                        f"{c['det']:.4f}", f"{c['f1']:.4f}"])
    print(f"\nСохранено: cifar_calibration.csv")

    # Сохранение калиброванной модели
    calib_data = {
        "keys": SBG_GREEDY_FEATURES,
        "pipeline": pipe,
        "threshold": float(thr),
        "target_fpr": target_fpr,
        "fpr_test": fpr_test_tuned,
        "det_test": det_test_tuned,
        "auc_test": auc_test,
    }
    with open("sentinel_q_cifar_v2_calibrated.pkl", "wb") as f:
        pickle.dump(calib_data, f)
    print(f"Сохранено: sentinel_q_cifar_v2_calibrated.pkl")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ROC curve
        ax = axes[0]
        from sklearn.metrics import roc_curve
        fpr_arr, tpr_arr, _ = roc_curve(
            np.concatenate([np.zeros(len(prob_test_b)),
                            np.ones(len(prob_test_a))]),
            np.concatenate([prob_test_b, prob_test_a]),
        )
        ax.plot(fpr_arr, tpr_arr, color="crimson", linewidth=2,
                label=f"AUC = {auc_test:.4f}")
        ax.scatter([fpr_05], [det_05], color="blue",
                   s=100, label=f"baseline (thr=0.5)", zorder=5)
        ax.scatter([fpr_test_tuned], [det_test_tuned], color="green",
                   s=100, label=f"tuned (FPR={target_fpr})", zorder=5)
        ax.axvline(0.05, color="orange", linestyle="--",
                   alpha=0.5, label="target FPR 5%")
        ax.set_xlabel("FPR")
        ax.set_ylabel("Detection rate")
        ax.set_title("ROC: baseline vs tuned")
        ax.legend()
        ax.grid(alpha=0.3)

        # Threshold curve
        ax = axes[1]
        thrs = [c["thr"] for c in curve]
        fprs = [c["fpr"] for c in curve]
        dets = [c["det"] for c in curve]
        ax.plot(thrs, fprs, "o-", color="blue", label="FPR")
        ax.plot(thrs, dets, "s-", color="crimson", label="Detection")
        ax.axvline(thr, color="green", linestyle="--",
                   label=f"tuned = {thr:.3f}")
        ax.set_xlabel("threshold")
        ax.set_ylabel("rate")
        ax.set_title("FPR / Detection vs threshold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("cifar_calibration.png", dpi=120)
        print("Сохранено: cifar_calibration.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()
    run(n_samples=args.n, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()