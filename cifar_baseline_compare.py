"""
cifar_baseline_compare.py — честное сравнение с baseline детекторами.

Сравниваем 4 метода на одном протоколе:
  1. Наш метод: bit-features + GradientBoosting
  2. LID (Ma et al., 2018) — Local Intrinsic Dimensionality
  3. Feature Squeezing (Xu et al., 2018)
  4. Mahalanobis (Lee et al., 2018)

Протокол:
  - Одна модель SmallCNN32 для всех
  - Train: 300 benign + 300 PGD-20
  - Calib: 100 benign (для threshold)
  - Test:  100 benign + 100 adversarial
  - Метрики: AUC, Detection@FPR=5%, Detection@FPR=10%

Честное замечание:
  LID, FS, Mahalanobis требуют доступа к модели (логиты/активации).
  Наш метод — тоже (PGD-примеры для обучения).
  Все методы сравниваются в равных условиях.

Запуск:
    python cifar_baseline_compare.py --n_benign 500
    python cifar_baseline_compare.py --n_benign 500 --attack pgd
    python cifar_baseline_compare.py --n_benign 500 --attack fgsm
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

    def features(self, x):
        """Возвращает активации перед последним слоем."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


# ── Bit-признаки (наш метод) ────────────────────────────────

def bit_features_one(img_np):
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


def extract_bit_features(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return bit_features_one(img)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(one, [arr[i] for i in range(len(arr))]))


def get_bit_feature_keys():
    return (
        ["frag", "frag_b", "H", "nu"]
        + [f"frag_bit{i}" for i in range(8)]
        + [f"H_bit{i}" for i in range(8)]
        + [f"td_bit{i}" for i in range(8)]
        + [f"hdiff_bit{i}" for i in range(8)]
        + [f"vdiff_bit{i}" for i in range(8)]
    )


# ── Атаки ───────────────────────────────────────────────────

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


def fgsm_attack(model, x, y, eps=0.05):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach().clamp(0, 1)


# ── Загрузка ────────────────────────────────────────────────

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


# ── Baseline 1: LID ─────────────────────────────────────────

def lid_score(model, x, y, k=20, batch_size=64):
    """
    Local Intrinsic Dimensionality.

    Для каждой точки x_i:
      1. Вычисляем активации в скрытом слое
      2. Находим k ближайших соседей в train-наборе
      3. LID = -1 / mean(log(d_i / d_k))
      4. Чем выше LID — тем более аномальная точка
    """
    from sklearn.neighbors import NearestNeighbors

    # Train-активации (на benign train)
    with torch.no_grad():
        train_acts = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(DEVICE)
            acts = model.features(batch).cpu().numpy()
            train_acts.append(acts)
        train_acts = np.vstack(train_acts)

    # kNN на train-активациях
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(train_acts)

    # Scores для входных данных
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(DEVICE)
            acts = model.features(batch).cpu().numpy()

            # Расстояния до k+1 соседей (первый — сама точка, если она в train)
            dists, _ = nn.kneighbors(acts, n_neighbors=k + 1)
            # dists[:, 0] может быть 0 (сама точка)
            # Используем dists[:, 1:]
            d = dists[:, 1:] + 1e-10  # (N, k)

            # LID = -1 / mean(log(d_i / d_k))
            d_k = d[:, -1:].repeat(k, axis=1)
            log_ratios = np.log(d / d_k)
            lid = -1.0 / (log_ratios.mean(axis=1) + 1e-10)
            scores.append(lid)

    return np.concatenate(scores)


# ── Baseline 2: Feature Squeezing ───────────────────────────

def feature_squeezing_score(model, x, batch_size=64,
                              bit_depth=4, use_median=False):
    """
    Feature Squeezing (Xu et al., 2018).

    1. Оригинальные логиты модели на x
    2. "Сжатые" логиты: bit_depth reduction (и/или median filter)
    3. Score = L1(softmax(orig) - softmax(squeezed))

    Чем выше score — тем больше модель "меняет мнение" при сжатии → атака.
    """
    scores = []

    def squeeze(images, depth=bit_depth, median=use_median):
        """Применяет сжатие к батчу."""
        x_sq = images.clone()

        if depth < 8:
            n_levels = 2 ** depth
            x_sq = torch.round(x_sq * (n_levels - 1)) / (n_levels - 1)

        if median:
            # Median filter 2x2 (approximation of the original paper's 3x3)
            x_sq = F.avg_pool2d(x_sq, kernel_size=2, stride=1, padding=0)
            # Resize back
            x_sq = F.interpolate(x_sq, size=images.shape[-2:],
                                  mode='bilinear', align_corners=False)

        return x_sq.clamp(0, 1)

    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = x[i:i + batch_size].to(DEVICE)

            logits_orig = model(batch)
            x_sq = squeeze(batch)
            logits_sq = model(x_sq)

            p_orig = F.softmax(logits_orig, dim=1)
            p_sq = F.softmax(logits_sq, dim=1)

            l1 = (p_orig - p_sq).abs().sum(dim=1)
            scores.append(l1.cpu().numpy())

    return np.concatenate(scores)


# ── Baseline 3: Mahalanobis ────────────────────────────────

def mahalanobis_score(model, x_train_b, y_train_b,
                       x_query, batch_size=64):
    """
    Mahalanobis Distance (Lee et al., 2018).

    1. Для каждого класса c: вычисляем mean μ_c и covariance Σ
       в пространстве признаков модели на train.
    2. Score(x) = min_c (f(x) - μ_c)^T Σ^{-1} (f(x) - μ_c)

    Чем больше — тем дальше от всех классов → аномалия.
    """
    # Train features
    with torch.no_grad():
        train_feats = []
        for i in range(0, len(x_train_b), batch_size):
            batch = x_train_b[i:i + batch_size].to(DEVICE)
            feats = model.features(batch).cpu().numpy()
            train_feats.append(feats)
        train_feats = np.vstack(train_feats)

    train_labels = y_train_b.cpu().numpy()
    n_classes = 10
    feat_dim = train_feats.shape[1]

    # Class means
    means = np.zeros((n_classes, feat_dim))
    for c in range(n_classes):
        mask = train_labels == c
        if mask.sum() > 0:
            means[c] = train_feats[mask].mean(axis=0)

    # Shared covariance (pooled)
    centered = np.zeros_like(train_feats)
    for c in range(n_classes):
        mask = train_labels == c
        centered[mask] = train_feats[mask] - means[c]

    cov = (centered.T @ centered) / max(len(train_feats) - n_classes, 1)
    # Regularization для численной устойчивости
    cov += np.eye(feat_dim) * 1e-4

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    # Scores для query
    scores = []
    with torch.no_grad():
        for i in range(0, len(x_query), batch_size):
            batch = x_query[i:i + batch_size].to(DEVICE)
            feats = model.features(batch).cpu().numpy()

            # Расстояния до каждого класса
            dists = np.zeros((len(feats), n_classes))
            for c in range(n_classes):
                diff = feats - means[c]
                # (x - μ)^T Σ^{-1} (x - μ)
                dists[:, c] = np.einsum(
                    'ij,jk,ik->i', diff, cov_inv, diff
                )

            # Min по классам
            scores.append(dists.min(axis=1))

    return np.concatenate(scores)


# ── Утилиты оценки ──────────────────────────────────────────

def evaluate_scores(scores_benign_calib, scores_benign_test,
                     scores_adv_test):
    """
    Универсальная оценка: AUC + Detection при FPR=5%/10%.
    """
    from sklearn.metrics import roc_auc_score

    y_test = np.concatenate([np.zeros(len(scores_benign_test)),
                              np.ones(len(scores_adv_test))])
    scores_test = np.concatenate([scores_benign_test, scores_adv_test])

    auc = roc_auc_score(y_test, scores_test)

    # Порог под FPR=5% на calib
    sorted_calib = np.sort(scores_benign_calib)[::-1]
    idx5 = min(int(0.05 * len(sorted_calib)), len(sorted_calib) - 1)
    thr5 = sorted_calib[idx5]
    fpr5 = float((scores_benign_test > thr5).mean())
    det5 = float((scores_adv_test > thr5).mean())

    idx10 = min(int(0.10 * len(sorted_calib)), len(sorted_calib) - 1)
    thr10 = sorted_calib[idx10]
    fpr10 = float((scores_benign_test > thr10).mean())
    det10 = float((scores_adv_test > thr10).mean())

    return {
        "auc": float(auc),
        "fpr5": fpr5, "det5": det5, "thr5": float(thr5),
        "fpr10": fpr10, "det10": det10, "thr10": float(thr10),
    }


# ── Главный эксперимент ─────────────────────────────────────

def run(n_benign=500, attack="pgd", n_workers=4):
    from sklearn.ensemble import GradientBoostingClassifier

    print("=" * 80)
    print(f"Baseline comparison — CIFAR-10, attack={attack}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    model, x_all, y_all = load_all()
    total = len(x_all)

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
    print(f"\n[1] Генерация {attack.upper()}-атаки ...")
    t0 = time.time()
    attack_fn = pgd_attack if attack == "pgd" else fgsm_attack

    x_a_train = attack_fn(model, x_b_train.clone(), y_b_train)
    x_a_calib = attack_fn(model, x_b_calib.clone(), y_b_calib)
    x_a_test = attack_fn(model, x_b_test.clone(), y_b_test)
    print(f"    {time.time()-t0:.1f}s")

    with torch.no_grad():
        asr = (model(x_b_test).argmax(1)
                != model(x_a_test).argmax(1)).float().mean().item()
    print(f"    ASR test = {asr:.3f}")

    # ── Метод 1: наш (bit features + GB)
    print(f"\n[2] Метод 1: bit-features + GradientBoosting")
    t0 = time.time()
    fb_train = extract_bit_features(x_b_train, n_workers)
    fa_train = extract_bit_features(x_a_train, n_workers)
    fb_calib = extract_bit_features(x_b_calib, n_workers)
    fb_test = extract_bit_features(x_b_test, n_workers)
    fa_test = extract_bit_features(x_a_test, n_workers)

    keys = get_bit_feature_keys()

    def to_X(feats):
        return np.array([[f[k] for k in keys] for f in feats])

    X_train = np.vstack([to_X(fb_train), to_X(fa_train)])
    y_train = np.concatenate([np.zeros(len(fb_train)),
                               np.ones(len(fa_train))])

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Scores = probability of "adversarial"
    sc_b_calib = clf.predict_proba(to_X(fb_calib))[:, 1]
    sc_b_test = clf.predict_proba(to_X(fb_test))[:, 1]
    sc_a_test = clf.predict_proba(to_X(fa_test))[:, 1]

    metrics_bit = evaluate_scores(sc_b_calib, sc_b_test, sc_a_test)
    print(f"    AUC = {metrics_bit['auc']:.4f}, "
          f"det@5% = {metrics_bit['det5']:.3f}, "
          f"det@10% = {metrics_bit['det10']:.3f}  "
          f"({time.time()-t0:.1f}s)")

    # ── Метод 2: LID
    print(f"\n[3] Метод 2: LID (Ma et al., 2018)")
    t0 = time.time()
    try:
        sc_b_calib = lid_score(model, x_b_calib)
        sc_b_test = lid_score(model, x_b_test)
        sc_a_test = lid_score(model, x_a_test)
        # LID: направление score —"чем больше, тем аномальнее"
        metrics_lid = evaluate_scores(sc_b_calib, sc_b_test, sc_a_test)
        print(f"    AUC = {metrics_lid['auc']:.4f}, "
              f"det@5% = {metrics_lid['det5']:.3f}, "
              f"det@10% = {metrics_lid['det10']:.3f}  "
              f"({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    Ошибка: {e}")
        metrics_lid = {"auc": 0.5, "det5": 0, "det10": 0}

    # ── Метод 3: Feature Squeezing
    print(f"\n[4] Метод 3: Feature Squeezing (Xu et al., 2018)")
    t0 = time.time()
    try:
        sc_b_calib = feature_squeezing_score(model, x_b_calib)
        sc_b_test = feature_squeezing_score(model, x_b_test)
        sc_a_test = feature_squeezing_score(model, x_a_test)
        metrics_fs = evaluate_scores(sc_b_calib, sc_b_test, sc_a_test)
        print(f"    AUC = {metrics_fs['auc']:.4f}, "
              f"det@5% = {metrics_fs['det5']:.3f}, "
              f"det@10% = {metrics_fs['det10']:.3f}  "
              f"({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    Ошибка: {e}")
        metrics_fs = {"auc": 0.5, "det5": 0, "det10": 0}

    # ── Метод 4: Mahalanobis
    print(f"\n[5] Метод 4: Mahalanobis (Lee et al., 2018)")
    t0 = time.time()
    try:
        sc_b_calib = mahalanobis_score(model, x_b_train, y_b_train,
                                         x_b_calib)
        sc_b_test = mahalanobis_score(model, x_b_train, y_b_train,
                                        x_b_test)
        sc_a_test = mahalanobis_score(model, x_b_train, y_b_train,
                                        x_a_test)
        metrics_md = evaluate_scores(sc_b_calib, sc_b_test, sc_a_test)
        print(f"    AUC = {metrics_md['auc']:.4f}, "
              f"det@5% = {metrics_md['det5']:.3f}, "
              f"det@10% = {metrics_md['det10']:.3f}  "
              f"({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"    Ошибка: {e}")
        metrics_md = {"auc": 0.5, "det5": 0, "det10": 0}

    # ── Итог
    print(f"\n{'=' * 80}")
    print(f"ИТОГ (attack = {attack}, ASR = {asr:.3f})")
    print(f"{'=' * 80}")
    print(f"\n  {'метод':<32} {'AUC':>8} {'det@5%':>8} {'det@10%':>8}")
    print("  " + "-" * 58)

    all_metrics = [
        ("Bit features + GB (our)", metrics_bit),
        ("LID (Ma et al., 2018)", metrics_lid),
        ("Feature Squeezing (Xu et al., 2018)", metrics_fs),
        ("Mahalanobis (Lee et al., 2018)", metrics_md),
    ]

    for name, m in all_metrics:
        marker = ""
        if m["auc"] >= 0.95:
            marker = " ✓✓✓"
        elif m["auc"] >= 0.90:
            marker = " ✓✓"
        elif m["auc"] >= 0.85:
            marker = " ✓"
        print(f"  {name:<32} {m['auc']:>8.4f} {m['det5']:>8.3f} "
              f"{m['det10']:>8.3f}{marker}")

    # Ранжирование
    sorted_by_auc = sorted(all_metrics, key=lambda x: -x[1]["auc"])
    print(f"\n  Ранжирование по AUC:")
    for i, (name, m) in enumerate(sorted_by_auc):
        print(f"    {i+1}. {name:<32} AUC = {m['auc']:.4f}")

    # Где наш метод
    our_rank = next(i for i, (n, _) in enumerate(sorted_by_auc)
                     if "our" in n.lower())
    if our_rank == 0:
        print(f"\n  ✓ Наш метод на первом месте")
    else:
        winner = sorted_by_auc[0][0]
        print(f"\n  ✓ Наш метод на {our_rank+1}-м месте, "
              f"лучший — {winner}")

    # ── Сохранение
    out = {
        "attack": attack, "asr": asr,
        "n_train": n_train, "n_calib": n_calib, "n_test": n_test,
        "results": {
            "our_bit_gb": metrics_bit,
            "lid": metrics_lid,
            "feature_squeezing": metrics_fs,
            "mahalanobis": metrics_md,
        },
    }
    with open(f"baseline_compare_{attack}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "auc", "det5", "det10",
                    "fpr5", "fpr10"])
        for name, m in all_metrics:
            w.writerow([name, f"{m['auc']:.4f}",
                        f"{m['det5']:.4f}", f"{m['det10']:.4f}",
                        f"{m['fpr5']:.4f}", f"{m['fpr10']:.4f}"])
    print(f"\n  Сохранено: baseline_compare_{attack}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--attack", default="pgd",
                    choices=["pgd", "fgsm"])
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    run(n_benign=args.n_benign, attack=args.attack,
        n_workers=args.jobs)


if __name__ == "__main__":
    main()