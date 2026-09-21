"""
cifar_anomaly.py — детекция аномалий без adversarial-примеров.

Идея: обучить модель ТОЛЬКО на benign данных. Adversarial — это
отклонения от benign-распределения. Никаких PGD при обучении.

Методы:
  1. Mahalanobis distance — расстояние до benign-центроида
  2. One-Class SVM — граница вокруг benign-облака
  3. Isolation Forest — изоляция выбросов
  4. Gaussian Mixture — вероятностная модель benign
  5. KNN-distance — расстояние до k-го соседа в benign
  6. Local Outlier Factor (LOF)

Плюс: calibrated threshold через квантили benign-оценок.

Запуск:
    python cifar_anomaly.py --n 1000
    python cifar_anomaly.py --n 1000 --target_fpr 0.05
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


# ── Anomaly Detectors ──────────────────────────────────────

def make_detectors():
    """Возвращает dict {name: factory_fn}."""
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import (LocalOutlierFactor,
                                     NearestNeighbors)
    from sklearn.covariance import EmpiricalCovariance

    def make_mahalanobis():
        return EmpiricalCovariance()

    def make_ocsvm():
        return OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")

    def make_iforest():
        return IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=42, n_jobs=-1,
        )

    def make_gmm():
        return GaussianMixture(
            n_components=5, covariance_type="full",
            random_state=42, max_iter=200,
        )

    def make_lof():
        return LocalOutlierFactor(
            n_neighbors=20, novelty=True, contamination=0.05,
        )

    return {
        "Mahalanobis": (make_mahalanobis, "higher"),
        "OneClassSVM": (make_ocsvm, "lower"),
        "IsolationForest": (make_iforest, "lower"),
        "GMM": (make_gmm, "lower"),
        "LOF": (make_lof, "lower"),
    }


def score_samples(detector, X, name):
    """
    Универсальный score: возвращает «чем выше, тем более аномально».
    """
    if name == "Mahalanobis":
        # squared Mahalanobis distance
        return detector.mahalanobis(X)
    if name == "GMM":
        # negative log-likelihood
        return -detector.score_samples(X)
    if name == "OneClassSVM":
        # decision_function: > 0 = inlier
        return -detector.decision_function(X)
    if name == "IsolationForest":
        # score_samples: < 0 = outlier
        return -detector.score_samples(X)
    if name == "LOF":
        return -detector.decision_function(X)
    raise ValueError(name)


# ── Основной эксперимент ────────────────────────────────────

def run(n_samples=1000, target_fpr=0.05):
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score

    print("=" * 78)
    print(f"Детекция аномалий на CIFAR-10 (без adversarial-примеров)")
    print(f"Device: {DEVICE}, target FPR = {target_fpr}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    n = min(n_samples, len(x_all))
    x_b = x_all[:n].to(DEVICE)
    y = y_all[:n].to(DEVICE)
    print(f"\nBenign: {n} примеров")

    # Split: 60% train / 20% val / 20% test
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    x_train = x_b[train_idx]; y_train = y[train_idx]
    x_val = x_b[val_idx]; y_val = y[val_idx]
    x_test = x_b[test_idx]; y_test = y[test_idx]
    print(f"    Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # Генерация PGD (ТОЛЬКО для оценки, не для обучения)
    print(f"\n[1] Генерация PGD-20 (для оценки, НЕ для обучения) ...")
    t0 = time.time()
    x_train_adv = pgd_attack(model, x_train, y_train)
    x_val_adv = pgd_attack(model, x_val, y_val)
    x_test_adv = pgd_attack(model, x_test, y_test)
    print(f"    {time.time()-t0:.1f}s")

    # ── Извлечение признаков
    print(f"\n[2] Извлечение bit-признаков ...")
    t0 = time.time()

    def extract(x_tensor):
        arr = x_tensor.cpu().numpy()
        return np.array([[real_bit_features(arr[i])[k]
                            for k in SBG_GREEDY_FEATURES]
                           for i in range(len(arr))])

    X_train_b = extract(x_train)
    X_val_b = extract(x_val)
    X_test_b = extract(x_test)
    X_test_a = extract(x_test_adv)
    print(f"    {time.time()-t0:.1f}s")

    # Нормализация на benign-train
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_b)
    X_val_scaled = scaler.transform(X_val_b)
    X_test_b_scaled = scaler.transform(X_test_b)
    X_test_a_scaled = scaler.transform(X_test_a)

    # ── Обучение детекторов на benign train
    print(f"\n[3] Обучение детекторов на benign train ({n_train} примеров) ...")
    detectors = make_detectors()
    results = []

    print(f"\n    {'detector':<18} {'val_thr':>10} {'test_FPR':>10} "
          f"{'test_det':>10} {'AUC':>8}")
    print("    " + "-" * 60)

    for name, (factory, direction) in detectors.items():
        t0 = time.time()
        det = factory()
        det.fit(X_train_scaled)

        # Scores
        scores_val_b = score_samples(det, X_val_scaled, name)
        scores_test_b = score_samples(det, X_test_b_scaled, name)
        scores_test_a = score_samples(det, X_test_a_scaled, name)

        # Threshold из benign val по квантилю
        thr = np.quantile(scores_val_b, 1 - target_fpr)

        # Метрики
        fpr = float((scores_test_b > thr).mean())
        det_rate = float((scores_test_a > thr).mean())

        y_true = np.concatenate([np.zeros(len(scores_test_b)),
                                  np.ones(len(scores_test_a))])
        scores_all = np.concatenate([scores_test_b, scores_test_a])
        try:
            auc = float(roc_auc_score(y_true, scores_all))
        except Exception:
            auc = 0.5

        results.append({
            "name": name, "thr": float(thr),
            "fpr": fpr, "det": det_rate, "auc": auc,
            "time": time.time() - t0,
        })
        print(f"    {name:<18} {thr:>10.3f} {fpr:>10.3f} "
              f"{det_rate:>10.3f} {auc:>8.4f}")

    # ── Ансамбль: средний rank
    print(f"\n[4] Ансамбль (rank-усреднение):")
    ranks_val_b = np.zeros(len(X_val_scaled))
    ranks_test_b = np.zeros(len(X_test_b_scaled))
    ranks_test_a = np.zeros(len(X_test_a_scaled))

    for name, (factory, _) in detectors.items():
        det = factory()
        det.fit(X_train_scaled)
        v = score_samples(det, X_val_scaled, name)
        tb = score_samples(det, X_test_b_scaled, name)
        ta = score_samples(det, X_test_a_scaled, name)
        # Нормализация к [0, 1] через квантиль
        rank_v = np.argsort(np.argsort(v)) / len(v)
        rank_tb = np.argsort(np.argsort(tb)) / len(tb)
        rank_ta = np.argsort(np.argsort(ta)) / len(ta)
        ranks_val_b += rank_v
        ranks_test_b += rank_tb
        ranks_test_a += rank_ta

    ranks_val_b /= len(detectors)
    ranks_test_b /= len(detectors)
    ranks_test_a /= len(detectors)

    thr_ens = np.quantile(ranks_val_b, 1 - target_fpr)
    fpr_ens = float((ranks_test_b > thr_ens).mean())
    det_ens = float((ranks_test_a > thr_ens).mean())

    y_true = np.concatenate([np.zeros(len(ranks_test_b)),
                              np.ones(len(ranks_test_a))])
    auc_ens = float(roc_auc_score(
        y_true, np.concatenate([ranks_test_b, ranks_test_a])
    ))
    print(f"    Threshold = {thr_ens:.3f}")
    print(f"    FPR = {fpr_ens:.3f}, Detection = {det_ens:.3f}, "
          f"AUC = {auc_ens:.4f}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    all_results = results + [{
        "name": "Ensemble", "thr": float(thr_ens),
        "fpr": fpr_ens, "det": det_ens, "auc": auc_ens,
    }]

    # Сортировка по F1-like
    def f1_like(r):
        p = r["det"]
        n = 1 - r["fpr"]
        return 2 * p * n / max(p + n, 1e-9)

    all_results.sort(key=lambda r: -f1_like(r))
    print(f"\n  {'detector':<18} {'FPR':>8} {'Detection':>10} "
          f"{'AUC':>8} {'F1-like':>10}")
    print("  " + "-" * 58)
    for r in all_results:
        print(f"  {r['name']:<18} {r['fpr']:>8.3f} {r['det']:>10.3f} "
              f"{r['auc']:>8.4f} {f1_like(r):>10.3f}")

    best = all_results[0]
    print(f"\n  Лучший: {best['name']}")
    print(f"    FPR = {best['fpr']:.3f}  Detection = {best['det']:.3f}  "
          f"AUC = {best['auc']:.4f}")

    if best["fpr"] <= 0.05 and best["det"] >= 0.75:
        print(f"\n✓✓ ЦЕЛЬ ДОСТИГНУТА: FPR ≤ 5%, Detection ≥ 75%")
    elif best["fpr"] <= 0.10 and best["det"] >= 0.70:
        print(f"\n✓ Хорошо: FPR ≤ 10%, Detection ≥ 70%")
    else:
        print(f"\n~ FPR / Detection всё ещё не идеальны")

    # Сохранение
    with open("cifar_anomaly_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "threshold", "fpr", "detection", "auc"])
        for r in all_results:
            w.writerow([r["name"], f"{r['thr']:.4f}",
                        f"{r['fpr']:.4f}", f"{r['det']:.4f}",
                        f"{r['auc']:.4f}"])
    print(f"\nСохранено: cifar_anomaly_results.csv")

    # Сохранение лучшей модели
    best_name = best["name"]
    if best_name == "Ensemble":
        save_data = {
            "type": "ensemble",
            "detectors": detectors,
            "scaler": scaler,
            "threshold": float(thr_ens),
            "keys": SBG_GREEDY_FEATURES,
            "fpr": fpr_ens, "det": det_ens, "auc": auc_ens,
        }
    else:
        factory, _ = detectors[best_name]
        det = factory()
        det.fit(X_train_scaled)
        save_data = {
            "type": best_name,
            "detector": det,
            "scaler": scaler,
            "threshold": best["thr"],
            "keys": SBG_GREEDY_FEATURES,
            "fpr": best["fpr"], "det": best["det"], "auc": best["auc"],
        }
    with open("sentinel_q_cifar_anomaly.pkl", "wb") as f:
        pickle.dump(save_data, f)
    print(f"Сохранено: sentinel_q_cifar_anomaly.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()
    run(n_samples=args.n, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()