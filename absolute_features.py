"""
absolute_features.py — признаки, не требующие оригинала.

Открытие: amplitude features (L2, L∞) с оригиналом дают утечку.
В реальности оригинала нет. Нужны АБСОЛЮТНЫЕ признаки.

Признаки:
  - TV (Total Variation)
  - Edge density (Sobel)
  - Local variance (скользящее окно 3×3)
  - DCT-статистика (энергия высоких частот)
  - Laplacian energy
  - Compression residual (медианный фильтр)

Гипотеза: FAB создаёт минимальные возмущения, но
изменяет локальные статистики (edges, variance).
Абсолютные признаки должны поймать FAB.

Запуск:
    python absolute_features.py --n_benign 500
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


# ── АБСОЛЮТНЫЕ признаки (без оригинала) ─────────────────────

def absolute_features_one(img_np):
    """
    img_np: (C, H, W) float [0, 1].

    Все признаки вычисляются БЕЗ знания оригинала.
    """
    out = {}

    # ── Bit-признаки (44)
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

    # ── АБСОЛЮТНЫЕ признаки
    # Total Variation по каждому каналу
    tv_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :]).mean()
    tv_w = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1]).mean()
    out["tv_h"] = float(tv_h)
    out["tv_w"] = float(tv_w)
    out["tv_total"] = float(tv_h + tv_w)

    # Edge density (Sobel approximation)
    # Horizontal gradient magnitude
    gx = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1])
    gy = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
    # Средний градиент
    out["grad_h_mean"] = float(gx.mean())
    out["grad_v_mean"] = float(gy.mean())
    # Максимальный градиент
    out["grad_h_max"] = float(gx.max())
    out["grad_v_max"] = float(gy.max())
    # Доля "сильных" edges (> 2/255)
    out["edge_density_h"] = float((gx > 2.0 / 255).mean())
    out["edge_density_v"] = float((gy > 2.0 / 255).mean())

    # Local variance (3×3 окно, через avg_pool)
    t = torch.from_numpy(img_np).float()
    t = t.unsqueeze(0)  # (1, C, H, W)
    mean_local = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    sq_local = F.avg_pool2d(t * t, kernel_size=3, stride=1, padding=1)
    var_local = (sq_local - mean_local ** 2).clamp(min=0)
    out["local_var_mean"] = float(var_local.mean())
    out["local_var_max"] = float(var_local.max())
    out["local_var_std"] = float(var_local.std())

    # Laplacian energy — применяем к каждому каналу отдельно
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3)
    C = img_np.shape[0]
    # Размножаем ядро для C каналов, groups=C — каждый канал отдельно
    lap_kernel_c = lap_kernel.repeat(C, 1, 1, 1)
    lap = F.conv2d(t, lap_kernel_c, padding=1, groups=C)
    out["laplacian_energy"] = float((lap ** 2).mean())
    out["laplacian_max"] = float(lap.abs().max())

    # DCT-статистика: энергия в разных частотных полосах
    # Приближение через resize + разницу
    t_orig = t.squeeze(0)  # (C, H, W)
    # Low freq: усреднение 2×2
    low = F.avg_pool2d(t, kernel_size=2, stride=2)  # (1, C, 16, 16)
    # High freq: разница между оригиналом и upsampled low
    low_up = F.interpolate(low, size=(32, 32), mode='nearest')
    high_freq = (t - low_up).abs()
    out["hf_mean"] = float(high_freq.mean())
    out["hf_max"] = float(high_freq.max())
    out["hf_std"] = float(high_freq.std())
    # Отношение high/low энергии
    low_energy = float((low_up ** 2).mean())
    high_energy = float((high_freq ** 2).mean())
    out["hf_lf_ratio"] = high_energy / max(low_energy, 1e-9)

    # ── Statistical moments per channel
    for c in range(img_np.shape[0]):
        ch = img_np[c]
        out[f"ch{c}_mean"] = float(ch.mean())
        out[f"ch{c}_std"] = float(ch.std())
        out[f"ch{c}_skew"] = float(((ch - ch.mean()) ** 3).mean()
                                     / max(ch.std() ** 3, 1e-9))
        out[f"ch{c}_kurt"] = float(((ch - ch.mean()) ** 4).mean()
                                     / max(ch.std() ** 4, 1e-9) - 3)

    return out


def extract_absolute(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return absolute_features_one(img)

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


def threshold_for_fpr(prob_benign, target_fpr):
    sorted_probs = np.sort(prob_benign)[::-1]
    idx = min(int(target_fpr * len(sorted_probs)),
              len(sorted_probs) - 1)
    return float(sorted_probs[idx])


def run(n_benign=500, n_workers=4, eps=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 82)
    print(f"Absolute features: bit + TV + edges + local var + DCT")
    print(f"Device: {DEVICE}, eps = {eps}")
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

    # ── Атаки
    print(f"\n[1] Генерация AutoAttack ...")
    attack_ids = {
        "APGD-CE":  "apgd-ce",
        "APGD-DLR": "apgd-dlr",
        "FAB":      "fab",
        "Square":   "square",
    }

    attacks = {}
    for name, atk_id in attack_ids.items():
        adversary = AutoAttack(
            model, norm='Linf', eps=eps, version='custom',
            device=DEVICE, verbose=False,
            attacks_to_run=[atk_id],
        )
        t0 = time.time()
        x_train_adv = adversary.run_standard_evaluation(
            x_b_train, y_b_train, bs=100
        )
        x_test_adv = adversary.run_standard_evaluation(
            x_b_test, y_b_test, bs=100
        )
        with torch.no_grad():
            pred_b = model(x_b_test).argmax(1)
            pred_a = model(x_test_adv).argmax(1)
            asr = ((pred_b != pred_a)
                    & (pred_b == y_b_test)).float().mean().item()
        attacks[name] = {
            "train": x_train_adv, "test": x_test_adv, "asr": asr,
        }
        print(f"    {name:<10} ASR={asr:.3f}  ({time.time()-t0:.1f}s)")

    # ── Извлечение признаков (абсолютные!)
    print(f"\n[2] Извлечение АБСОЛЮТНЫХ признаков ...")
    t0 = time.time()
    fb_train = extract_absolute(x_b_train, n_workers)
    fb_test = extract_absolute(x_b_test, n_workers)
    fb_calib = extract_absolute(x_b_calib, n_workers)

    fa_train = {}
    fa_test = {}
    for name in attacks:
        fa_train[name] = extract_absolute(attacks[name]["train"], n_workers)
        fa_test[name] = extract_absolute(attacks[name]["test"], n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    bit_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("frag", "H_bit", "td_bit", "hdiff_bit", "vdiff_bit", "nu")
    )]
    abs_keys = [k for k in keys if k not in bit_keys]
    print(f"    Bit-признаков: {len(bit_keys)}")
    print(f"    Абсолютных: {len(abs_keys)}")
    print(f"    Всего: {len(keys)}")
    print(f"    Абсолютные: {abs_keys[:10]}...")

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    # ── Сравнение
    print(f"\n[3] Сравнение:")
    print(f"    {'attack':<10} {'bit-only':>10} {'abs-only':>10} "
          f"{'bit+abs':>10}")
    print("    " + "-" * 44)

    results = {}
    for name in attacks:
        row = {}

        for label, kset in [("bit", bit_keys), ("abs", abs_keys),
                              ("all", keys)]:
            X_tr = np.vstack([to_X(fb_train, kset),
                               to_X(fa_train[name], kset)])
            y_tr = np.concatenate([np.zeros(len(fb_train)),
                                    np.ones(len(fa_train[name]))])
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob_b = clf.predict_proba(to_X(fb_test, kset))[:, 1]
            prob_a = clf.predict_proba(to_X(fa_test[name], kset))[:, 1]
            y_test = np.concatenate([np.zeros(len(prob_b)),
                                      np.ones(len(prob_a))])
            auc = roc_auc_score(y_test, np.concatenate([prob_b, prob_a]))
            row[label] = auc

        results[name] = row
        marker = ""
        if row["all"] > 0.90:
            marker = " ✓✓✓"
        elif row["all"] > 0.80:
            marker = " ✓✓"
        print(f"    {name:<10} {row['bit']:>10.4f} {row['abs']:>10.4f} "
              f"{row['all']:>10.4f}{marker}")

    # ── Detection @ FPR=5%
    print(f"\n[4] Detection @ FPR=5%:")
    print(f"    {'attack':<10} {'bit-only':>10} {'abs-only':>10} "
          f"{'bit+abs':>10}")
    print("    " + "-" * 44)

    for name in attacks:
        dets = {}
        for label, kset in [("bit", bit_keys), ("abs", abs_keys),
                              ("all", keys)]:
            X_tr = np.vstack([to_X(fb_train, kset),
                               to_X(fa_train[name], kset)])
            y_tr = np.concatenate([np.zeros(len(fb_train)),
                                    np.ones(len(fa_train[name]))])
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob_b_calib = clf.predict_proba(
                to_X(fb_calib, kset))[:, 1]
            thr = threshold_for_fpr(prob_b_calib, 0.05)
            prob_a = clf.predict_proba(to_X(fa_test[name], kset))[:, 1]
            dets[label] = float((prob_a > thr).mean())

        marker = ""
        if dets["all"] > 0.85:
            marker = " ✓✓✓"
        print(f"    {name:<10} {dets['bit']:>10.3f} {dets['abs']:>10.3f} "
              f"{dets['all']:>10.3f}{marker}")

    # ── Топ-признаки для FAB
    print(f"\n[5] Топ-15 признаков для FAB (feature importance):")
    X_tr = np.vstack([to_X(fb_train, keys), to_X(fa_train["FAB"], keys)])
    y_tr = np.concatenate([np.zeros(len(fb_train)),
                            np.ones(len(fa_train["FAB"]))])
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(15):
        idx = sorted_idx[i]
        key = keys[idx]
        kind = "BIT" if key in bit_keys else "ABS"
        print(f"    {i+1:>2}. [{kind}] {key:<20} "
              f"importance = {importances[idx]:.4f}")

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    fab_bit = results["FAB"]["bit"]
    fab_abs = results["FAB"]["abs"]
    fab_all = results["FAB"]["all"]

    print(f"\n  FAB:")
    print(f"    bit-only:  AUC = {fab_bit:.4f}")
    print(f"    abs-only:  AUC = {fab_abs:.4f}")
    print(f"    bit+abs:   AUC = {fab_all:.4f}")

    if fab_all > 0.90:
        print(f"\n  ✓✓✓ АБСОЛЮТНЫЕ ПРИЗНАКИ РЕШАЮТ ПРОБЛЕМУ FAB")
        print(f"      Без оригинала, только структура изображения.")
        print(f"      AUC = {fab_all:.4f} (vs {fab_bit:.4f} для bit-only).")
        print(f"\n  Публикационное утверждение:")
        print(f"      «FAB-атака не детектируется bit-plane признаками,")
        print(f"       но детектируется абсолютными признаками (TV, edges,")
        print(f"       local variance, DCT) без доступа к оригиналу.»")
    elif fab_abs > 0.80:
        print(f"\n  ✓ Частичный успех: abs-only = {fab_abs:.4f}")
        print(f"      Улучшение, но не полное решение.")
    else:
        print(f"\n  ✗ Абсолютные признаки не решают проблему FAB")
        print(f"      AUC = {fab_abs:.4f}")

    with open("absolute_features_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "asr", "auc_bit", "auc_abs", "auc_all"])
        for name in attacks:
            r = results[name]
            w.writerow([name, f"{attacks[name]['asr']:.4f}",
                        f"{r['bit']:.4f}", f"{r['abs']:.4f}",
                        f"{r['all']:.4f}"])
    print(f"\n  Сохранено: absolute_features_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs, eps=args.eps)


if __name__ == "__main__":
    main()