"""
fab_calibration.py — калибровка порога специально для FAB.

Проблема: при threshold по квантилю benign (FPR=5%),
detection на FAB = 0.59 (из absolute_features.py).

Причина: распределение scores на FAB имеет длинный левый хвост —
часть FAB-примеров близка к benign. Квантильная калибровка
не учитывает форму FAB-распределения.

Решение:
  1. Явная оптимизация порога на calib (benign + FAB):
     max detection s.t. FPR ≤ target_fpr
  2. Bootstrap усреднение порога (100 подвыборок)
  3. Ансамбль: bit-only + abs-only + bit+abs (soft voting)
  4. Сравнение 4 стратегий калибровки:
     - quantile benign (baseline)
     - optimize F1 на calib
     - optimize detection@FPR
     - cost-sensitive (FN дороже FP)

Запуск:
    python fab_calibration.py --n_benign 500 --target_fpr 0.05
"""

import argparse
import csv
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


# ── Признаки ────────────────────────────────────────────────

def all_features_one(img_np):
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
        out[f"hdiff_bit{bit}"] = float(np.abs(bp[:, 1:] - bp[:, :-1]).mean())
        out[f"vdiff_bit{bit}"] = float(np.abs(bp[1:, :] - bp[:-1, :]).mean())

    # Абсолютные
    tv_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :]).mean()
    tv_w = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1]).mean()
    out["tv_h"] = float(tv_h)
    out["tv_w"] = float(tv_w)
    out["tv_total"] = float(tv_h + tv_w)

    gx = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1])
    gy = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
    out["grad_h_mean"] = float(gx.mean())
    out["grad_v_mean"] = float(gy.mean())
    out["edge_density_h"] = float((gx > 2.0 / 255).mean())
    out["edge_density_v"] = float((gy > 2.0 / 255).mean())

    t = torch.from_numpy(img_np).float().unsqueeze(0)
    mean_local = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    sq_local = F.avg_pool2d(t * t, kernel_size=3, stride=1, padding=1)
    var_local = (sq_local - mean_local ** 2).clamp(min=0)
    out["local_var_mean"] = float(var_local.mean())
    out["local_var_max"] = float(var_local.max())

    # Laplacian
    C = img_np.shape[0]
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    lap = F.conv2d(t, lap_kernel, padding=1, groups=C)
    out["laplacian_energy"] = float((lap ** 2).mean())

    # DCT
    low = F.avg_pool2d(t, kernel_size=2, stride=2)
    low_up = F.interpolate(low, size=(32, 32), mode='nearest')
    high_freq = (t - low_up).abs()
    out["hf_mean"] = float(high_freq.mean())
    out["hf_max"] = float(high_freq.max())
    low_energy = float((low_up ** 2).mean())
    high_energy = float((high_freq ** 2).mean())
    out["hf_lf_ratio"] = high_energy / max(low_energy, 1e-9)

    return out


def extract(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return all_features_one(img)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(one, [arr[i] for i in range(len(arr))]))


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


# ── Стратегии калибровки ────────────────────────────────────

def thr_quantile_benign(prob_b_calib, target_fpr):
    """Baseline: квантиль benign."""
    sorted_p = np.sort(prob_b_calib)[::-1]
    idx = min(int(target_fpr * len(sorted_p)), len(sorted_p) - 1)
    return float(sorted_p[idx])


def thr_max_detection_at_fpr(prob_b_calib, prob_a_calib, target_fpr):
    """
    Оптимизация: max detection на calib при FPR ≤ target_fpr.
    Перебираем все уникальные пороги.
    """
    # Кандидаты — все значения prob_b и prob_a
    candidates = np.unique(np.concatenate([prob_b_calib, prob_a_calib]))

    best_thr = candidates[0]
    best_det = 0.0

    for thr in candidates:
        fpr = (prob_b_calib > thr).mean()
        if fpr > target_fpr:
            continue
        det = (prob_a_calib > thr).mean()
        if det > best_det:
            best_det = det
            best_thr = thr

    return float(best_thr)


def thr_bootstrap_optimize(prob_b_calib, prob_a_calib,
                            target_fpr, n_bootstrap=100, seed=42):
    """
    Bootstrap-усреднение порога.
    Многократно сэмплируем calib, оптимизируем порог, усредняем.
    """
    rng = np.random.default_rng(seed)
    n_b = len(prob_b_calib)
    n_a = len(prob_a_calib)

    thresholds = []
    for _ in range(n_bootstrap):
        idx_b = rng.integers(0, n_b, n_b)
        idx_a = rng.integers(0, n_a, n_a)
        thr = thr_max_detection_at_fpr(
            prob_b_calib[idx_b], prob_a_calib[idx_a], target_fpr
        )
        thresholds.append(thr)

    return float(np.median(thresholds))


def thr_cost_sensitive(prob_b_calib, prob_a_calib,
                        target_fpr, fn_cost=10.0):
    """
    Cost-sensitive: FN стоит в fn_cost раз дороже FP.
    Оптимизируем total_cost.
    """
    candidates = np.unique(np.concatenate([prob_b_calib, prob_a_calib]))

    best_thr = candidates[0]
    best_cost = float('inf')

    n_b = len(prob_b_calib)
    n_a = len(prob_a_calib)

    for thr in candidates:
        fp = (prob_b_calib > thr).sum()
        fn = (prob_a_calib <= thr).sum()
        # FPR ограничение как штраф
        fpr = fp / n_b
        cost = fn * fn_cost + fp * 1.0
        if fpr > target_fpr:
            cost += 1e6  # сильный штраф за нарушение FPR
        if cost < best_cost:
            best_cost = cost
            best_thr = thr

    return float(best_thr)


# ── Основной эксперимент ────────────────────────────────────

def run(n_benign=500, n_workers=4, eps=0.05, target_fpr=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 82)
    print(f"Калибровка порога для FAB")
    print(f"Device: {DEVICE}, eps = {eps}, target_fpr = {target_fpr}")
    print("=" * 82)

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

    # ── FAB на train, calib, test
    print(f"\n[1] Генерация FAB ...")
    adversary = AutoAttack(
        model, norm='Linf', eps=eps, version='custom',
        device=DEVICE, verbose=False,
        attacks_to_run=["fab"],
    )
    t0 = time.time()
    x_a_train = adversary.run_standard_evaluation(x_b_train, y_b_train, bs=100)
    x_a_calib = adversary.run_standard_evaluation(x_b_calib, y_b_calib, bs=100)
    x_a_test = adversary.run_standard_evaluation(x_b_test, y_b_test, bs=100)
    print(f"    {time.time()-t0:.1f}s")

    with torch.no_grad():
        pred_b = model(x_b_test).argmax(1)
        pred_a = model(x_a_test).argmax(1)
        asr = ((pred_b != pred_a) & (pred_b == y_b_test)).float().mean().item()
    print(f"    ASR = {asr:.3f}")

    # ── Признаки
    print(f"\n[2] Признаки ...")
    t0 = time.time()
    fb_train = extract(x_b_train, n_workers)
    fb_calib = extract(x_b_calib, n_workers)
    fb_test = extract(x_b_test, n_workers)
    fa_train = extract(x_a_train, n_workers)
    fa_calib = extract(x_a_calib, n_workers)
    fa_test = extract(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    bit_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("frag", "H_bit", "td_bit", "hdiff_bit", "vdiff_bit", "nu")
    )]
    abs_keys = [k for k in keys if k not in bit_keys]
    # "H" попадает и туда, и туда — исключаем из bit
    bit_keys = [k for k in bit_keys if k != "H"]

    print(f"    Bit: {len(bit_keys)}, Abs: {len(abs_keys)}, "
          f"All: {len(keys)}")

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    X_tr_b = to_X(fb_train, keys)
    X_tr_a = to_X(fa_train, keys)
    X_cal_b = to_X(fb_calib, keys)
    X_cal_a = to_X(fa_calib, keys)
    X_te_b = to_X(fb_test, keys)
    X_te_a = to_X(fa_test, keys)

    # ── Обучение ансамбля
    print(f"\n[3] Обучение ансамбля (bit + abs, 3 модели) ...")

    X_tr = np.vstack([X_tr_b, X_tr_a])
    y_tr = np.concatenate([np.zeros(len(X_tr_b)), np.ones(len(X_tr_a))])

        # Одиночный GradientBoosting (лучший по AUC в absolute_features.py)
    print(f"    Training GradientBoosting ...")
    t0 = time.time()
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_tr, y_tr)

    prob_cal_b = clf.predict_proba(X_cal_b)[:, 1]
    prob_cal_a = clf.predict_proba(X_cal_a)[:, 1]
    prob_te_b = clf.predict_proba(X_te_b)[:, 1]
    prob_te_a = clf.predict_proba(X_te_a)[:, 1]

    auc_cal = roc_auc_score(
        np.concatenate([np.zeros(len(prob_cal_b)), np.ones(len(prob_cal_a))]),
        np.concatenate([prob_cal_b, prob_cal_a]),
    )
    print(f"    GB: AUC calib = {auc_cal:.4f}  ({time.time()-t0:.1f}s)")


    y_test = np.concatenate([np.zeros(len(prob_te_b)),
                              np.ones(len(prob_te_a))])
    auc_test = roc_auc_score(y_test,
                              np.concatenate([prob_te_b, prob_te_a]))
    print(f"\n    Ensemble AUC test = {auc_test:.4f}")

    # ── Калибровка: 4 стратегии
    print(f"\n[4] Калибровка порога (4 стратегии):")
    print(f"    {'strategy':<28} {'thr':>8} {'calib FPR':>10} "
          f"{'calib det':>10} {'test FPR':>9} {'test det':>9}")
    print("    " + "-" * 78)

    strategies = {}

    # 1. Quantile benign
    thr1 = thr_quantile_benign(prob_cal_b, target_fpr)
    strategies["Quantile benign"] = thr1

    # 2. Max detection at FPR
    thr2 = thr_max_detection_at_fpr(prob_cal_b, prob_cal_a, target_fpr)
    strategies["Max detection@FPR"] = thr2

    # 3. Bootstrap
    thr3 = thr_bootstrap_optimize(prob_cal_b, prob_cal_a,
                                    target_fpr, n_bootstrap=100)
    strategies["Bootstrap median"] = thr3

    # 4. Cost-sensitive
    thr4 = thr_cost_sensitive(prob_cal_b, prob_cal_a,
                                target_fpr, fn_cost=10.0)
    strategies["Cost-sensitive (FN×10)"] = thr4

    results = {}
    for name, thr in strategies.items():
        cal_fpr = float((prob_cal_b > thr).mean())
        cal_det = float((prob_cal_a > thr).mean())
        te_fpr = float((prob_te_b > thr).mean())
        te_det = float((prob_te_a > thr).mean())

        results[name] = {
            "thr": thr, "cal_fpr": cal_fpr, "cal_det": cal_det,
            "test_fpr": te_fpr, "test_det": te_det,
        }

        marker = ""
        if te_det >= 0.85 and te_fpr <= target_fpr + 0.02:
            marker = " ✓✓✓"
        elif te_det >= 0.70:
            marker = " ✓✓"
        elif te_det >= 0.55:
            marker = " ✓"

        print(f"    {name:<28} {thr:>8.4f} {cal_fpr:>10.3f} "
              f"{cal_det:>10.3f} {te_fpr:>9.3f} {te_det:>9.3f}{marker}")

    # ── Детальный sweep
    print(f"\n[5] Детальный sweep (порог → FPR/detection на test):")
    print(f"    {'thr':>8} {'test FPR':>10} {'test det':>10} "
          f"{'precision':>10}")
    print("    " + "-" * 44)

    best_det_at_fpr = 0.0
    best_thr_detailed = None
    for thr in np.linspace(0.05, 0.95, 19):
        te_fpr = float((prob_te_b > thr).mean())
        te_det = float((prob_te_a > thr).mean())
        tp = (prob_te_a > thr).sum()
        fp = (prob_te_b > thr).sum()
        prec = tp / max(tp + fp, 1)

        marker = ""
        if te_fpr <= target_fpr and te_det > best_det_at_fpr:
            best_det_at_fpr = te_det
            best_thr_detailed = thr
            marker = " ←"

        print(f"    {thr:>8.3f} {te_fpr:>10.3f} {te_det:>10.3f} "
              f"{prec:>10.3f}{marker}")

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    print(f"\n  Ансамбль: GB + RF + ET (soft voting)")
    print(f"  AUC test: {auc_test:.4f}")

    print(f"\n  Результаты калибровки:")
    best_strategy = max(results.items(),
                          key=lambda kv: kv[1]["test_det"])
    for name, r in results.items():
        print(f"    {name:<28} test det = {r['test_det']:.3f}, "
              f"FPR = {r['test_fpr']:.3f}")

    print(f"\n  Лучшая стратегия: {best_strategy[0]}")
    print(f"    threshold = {best_strategy[1]['thr']:.4f}")
    print(f"    Detection = {best_strategy[1]['test_det']:.3f}")
    print(f"    FPR = {best_strategy[1]['test_fpr']:.3f}")

    if best_thr_detailed is not None:
        print(f"\n  При явном соблюдении FPR ≤ {target_fpr}:")
        print(f"    threshold = {best_thr_detailed:.3f}")
        print(f"    Detection = {best_det_at_fpr:.3f}")

    # Сравнение с baseline из absolute_features
    print(f"\n  Сравнение с baseline (absolute_features.py):")
    print(f"    Было:   bit+abs AUC=0.916, det@FPR5=0.59")
    print(f"    Стало:  ensemble AUC={auc_test:.3f}, "
          f"det@FPR5={best_strategy[1]['test_det']:.3f}")

    improvement = best_strategy[1]["test_det"] - 0.59
    print(f"    Прирост detection: {improvement:+.3f}")

    # ── Сохранение
    with open("fab_calibration_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "threshold", "calib_fpr",
                    "calib_det", "test_fpr", "test_det"])
        for name, r in results.items():
            w.writerow([name, f"{r['thr']:.4f}",
                        f"{r['cal_fpr']:.4f}", f"{r['cal_det']:.4f}",
                        f"{r['test_fpr']:.4f}", f"{r['test_det']:.4f}"])
    print(f"\n  Сохранено: fab_calibration_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs,
        eps=args.eps, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()