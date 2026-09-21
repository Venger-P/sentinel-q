"""
multiscale_frag_v3.py — cross-compressor deltas + multiscale frag.

Идея: zlib (LZ77, локальные повторы) и bz2 (BWT, глобальная структура)
по-разному реагируют на adversarial-атаку. Разница между ними на каждом
масштабе может дать независимый признак.

Новые признаки:
  - d_frag_zb_32 = frag_z_32 - frag_b_32
  - d_frag_zb_16, d_frag_zb_8, d_frag_zb_4
  - ratio_frag_zb_32 = frag_z_32 / (frag_b_32 + eps)
  - + то же для lzma, если включён

Плюс сохраняются все признаки из v2 для сравнения.

Запуск:
    python multiscale_frag_v3.py --n 500
    python multiscale_frag_v3.py --n 500 --compressors zlib bz2 lzma
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy, n_unique_bytes


# ── Мультимасштабные признаки ───────────────────────────────

def image_to_bytes_at_scale(img_np, factor):
    """img_np: (C, H, W) float [0,1]."""
    if factor == 1:
        return (img_np * 255).astype(np.uint8).tobytes()

    t = torch.from_numpy(img_np).float()
    x = t.unsqueeze(0)
    pooled = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return (pooled.squeeze(0).numpy() * 255).astype(np.uint8).tobytes()


def multiscale_features(img_np, compressors=("zlib", "bz2")):
    """
    Полный набор признаков:
      - frag по каждому компрессору на каждом масштабе
      - усреднённый frag на каждом масштабе
      - cross-compressor deltas и ratios
      - multiscale deltas
      - H, n_unique на каждом масштабе
    """
    out = {}
    frags_by_scale = {}
    frag_by_comp_scale = {}  # (comp, scale) -> frag

    scales = [(1, "32"), (2, "16"), (4, "8"), (8, "4")]

    for factor, sname in scales:
        data = image_to_bytes_at_scale(img_np, factor)

        frags_list = []
        for algo in compressors:
            f = fragility(data, algo)
            frags_list.append(f)
            out[f"frag_{algo[0]}_{sname}"] = f
            frag_by_comp_scale[(algo, sname)] = f

        f_avg = float(np.mean(frags_list))
        out[f"frag_{sname}"] = f_avg
        frags_by_scale[sname] = f_avg

        out[f"H_{sname}"] = shannon_entropy(data)
        out[f"nu_{sname}"] = n_unique_bytes(data)

    # ── Cross-compressor deltas и ratios
    eps = 1e-6
    if len(compressors) >= 2:
        for i, c1 in enumerate(compressors):
            for c2 in compressors[i + 1:]:
                for sname in ["32", "16", "8", "4"]:
                    key1 = f"frag_{c1[0]}_{sname}"
                    key2 = f"frag_{c2[0]}_{sname}"
                    if key1 in out and key2 in out:
                        f1, f2 = out[key1], out[key2]
                        tag = f"{c1[0]}{c2[0]}"
                        out[f"d_frag_{tag}_{sname}"] = f1 - f2
                        out[f"ratio_{tag}_{sname}"] = (
                            f1 / (f2 + eps)
                        )

    # ── Multiscale deltas
    out["d_frag_32_16"] = frags_by_scale["32"] - frags_by_scale["16"]
    out["d_frag_16_8"]  = frags_by_scale["16"] - frags_by_scale["8"]
    out["d_frag_8_4"]   = frags_by_scale["8"]  - frags_by_scale["4"]
    out["frag_mean"]    = float(np.mean(list(frags_by_scale.values())))
    out["frag_std"]     = float(np.std(list(frags_by_scale.values())))
    out["frag_range"]   = (max(frags_by_scale.values())
                            - min(frags_by_scale.values()))

    return out


# ── Загрузка CIFAR ──────────────────────────────────────────

def load_cifar():
    cache = Path("../Kvorb/V5.0/adv_cache_cifar")
    if not cache.exists():
        cache = Path("adv_cache_cifar")
    if not cache.exists():
        raise SystemExit("adv_cache_cifar/ не найден")

    benign = np.load(cache / "benign.npz")
    adv = np.load(cache / "adversarial.npz")
    return benign["x"], benign["y"], adv["x"], adv["y"]


# ── SNR и AUC ───────────────────────────────────────────────

def snr_for_feature(feature_b, feature_a):
    delta = float(np.mean(feature_a) - np.mean(feature_b))
    std = float(np.std(feature_b))
    if std < 1e-9:
        return 0.0, delta
    return abs(delta) / std, delta


def auc_for_feature(feature_b, feature_a):
    n_b, n_a = len(feature_b), len(feature_a)
    all_vals = np.concatenate([feature_b, feature_a])
    order = np.argsort(all_vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(all_vals) + 1)
    rank_b = ranks[:n_b].sum()
    u = rank_b - n_b * (n_b + 1) / 2
    auc = u / (n_b * n_a)
    if np.mean(feature_a) < np.mean(feature_b):
        return auc
    return 1 - auc


# ── Основной эксперимент ────────────────────────────────────

def run(n_samples=500, compressors=("zlib", "bz2")):
    print("=" * 78)
    print("Multiscale frag v3 — cross-compressor deltas")
    print("=" * 78)
    print(f"Компрессоры: {compressors}")

    print(f"\n[1] Загрузка ...")
    x_b, y_b, x_a, y_a = load_cifar()
    n = min(n_samples, len(x_b), len(x_a))
    x_b = x_b[:n]
    x_a = x_a[:n]
    print(f"    {n} benign + {n} adversarial")

    print(f"\n[2] Признаки ...")
    t0 = time.time()
    feats_b, feats_a = [], []
    for i in range(n):
        feats_b.append(multiscale_features(x_b[i], compressors))
        feats_a.append(multiscale_features(x_a[i], compressors))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}  ({time.time()-t0:.1f}s)")

    keys = sorted(feats_b[0].keys())
    print(f"    Признаков на образец: {len(keys)}")

    # ── SNR для каждого
    print(f"\n[3] Top-20 признаков по SNR:")
    snr_results = []
    for key in keys:
        fb = np.array([f[key] for f in feats_b])
        fa = np.array([f[key] for f in feats_a])
        snr, delta = snr_for_feature(fb, fa)
        auc = auc_for_feature(fb, fa)
        snr_results.append({
            "key": key, "snr": snr, "auc": auc, "delta": delta,
            "mean_b": float(fb.mean()), "mean_a": float(fa.mean()),
        })

    sorted_by_snr = sorted(snr_results, key=lambda r: -r["snr"])
    header = (f"    {'feature':<22} {'benign':>11} {'adversarial':>12} "
              f"{'Δ':>10} {'SNR':>7} {'AUC':>7}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in sorted_by_snr[:20]:
        print(f"    {r['key']:<22} {r['mean_b']:>11.4f} "
              f"{r['mean_a']:>12.4f} {r['delta']:>+10.4f} "
              f"{r['snr']:>7.3f} {r['auc']:>7.3f}")

    # ── Baseline
    print(f"\n[4] Ключевые признаки для сравнения:")
    for key in ["frag_32", "frag_z_32", "frag_b_32",
                "d_frag_32_16", "d_frag_zb_32"]:
        r = next((r for r in snr_results if r["key"] == key), None)
        if r:
            print(f"    {key:<22} SNR = {r['snr']:.3f}, "
                  f"AUC = {r['auc']:.3f}")

    best = sorted_by_snr[0]
    print(f"\n[5] Best single: {best['key']}")
    print(f"    SNR = {best['snr']:.3f}, AUC = {best['auc']:.3f}")

    # ── LR
    print(f"\n[6] Logistic Regression:")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("    sklearn не установлен")
        return

    def make_xy(keys_list):
        X_b = np.array([[f[k] for k in keys_list] for f in feats_b])
        X_a = np.array([[f[k] for k in keys_list] for f in feats_a])
        return np.vstack([X_b, X_a]), np.concatenate(
            [np.zeros(n), np.ones(n)]
        )

    # Feature sets
    cross_comp_keys = [k for k in keys
                        if k.startswith("d_frag_") and len(k.split("_")) == 4
                        and k.split("_")[2] in ("zb", "zl", "bl")]
    cross_ratio_keys = [k for k in keys if k.startswith("ratio_")]

    frag_avg_keys = [f"frag_{s}" for s in ["32", "16", "8", "4"]]
    frag_z_keys = [f"frag_z_{s}" for s in ["32", "16", "8", "4"]]
    frag_b_keys = [f"frag_b_{s}" for s in ["32", "16", "8", "4"]]
    frag_l_keys = [f"frag_l_{s}" for s in ["32", "16", "8", "4"]
                   if f"frag_l_{s}" in keys]
    H_keys = [f"H_{s}" for s in ["32", "16", "8", "4"]]
    nu_keys = [f"nu_{s}" for s in ["32", "16", "8", "4"]]

    feature_sets = {
        "frag_32 (baseline)":          ["frag_32"],
        "all frags (avg)":             frag_avg_keys,
        "frags by compressor":         frag_z_keys + frag_b_keys + frag_l_keys,
        "cross-comp deltas only":      cross_comp_keys,
        "cross-comp ratios only":      cross_ratio_keys,
        "frags + cross deltas":        frag_z_keys + frag_b_keys + cross_comp_keys,
        "frags + H + nu":              frag_avg_keys + H_keys + nu_keys,
        "frags + H + cross deltas":    frag_z_keys + frag_b_keys + H_keys
                                        + cross_comp_keys,
        "everything":                  keys,
    }

    print(f"    {'features':<32} {'n_feat':>7} {'AUC (5-fold)':>16}")
    print("    " + "-" * 58)

    best_auc = 0
    best_name = None
    results_lr = []

    for name, keys_list in feature_sets.items():
        if not keys_list:
            continue
        X, y = make_xy(keys_list)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        auc = float(scores.mean())
        auc_std = float(scores.std())

        results_lr.append({
            "name": name, "auc": auc, "n_features": len(keys_list),
        })

        marker = ""
        if auc > best_auc:
            best_auc = auc
            best_name = name
            marker = " ←"

        print(f"    {name:<32} {len(keys_list):>7} "
              f"{auc:>10.4f} ± {auc_std:.4f}{marker}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    baseline = next((r for r in results_lr
                     if r["name"] == "frag_32 (baseline)"), None)

    print(f"\nBaseline (frag_32):  SNR = "
          f"{next(r['snr'] for r in snr_results if r['key']=='frag_32'):.3f}, "
          f"AUC = {baseline['auc'] if baseline else 0:.3f}")
    print(f"Best single:         {best['key']}, "
          f"SNR = {best['snr']:.3f}, AUC = {best['auc']:.3f}")
    print(f"Best LR:             {best_name}, AUC = {best_auc:.3f}")

    print(f"\nСравнение с историей:")
    print(f"  CIFAR v1 (zlib, frag_32):       AUC = 0.64")
    print(f"  CIFAR v2 (zlib+bz2):            AUC = 0.78")
    print(f"  CIFAR v2 (zlib+bz2+lzma):       AUC = 0.82")
    print(f"  CIFAR v3 (cross-comp deltas):   AUC = {best_auc:.3f}")

    if best_auc > 0.85:
        print(f"\n✓ Cross-compressor deltas РАБОТАЮТ")
        print(f"   AUC = {best_auc:.3f} — production-ready для CIFAR.")
    elif best_auc > 0.82:
        print(f"\n~ Улучшение над v2: AUC = {best_auc:.3f}")
        print(f"   Дополнительный прирост от cross-compressor признаков.")
    elif best_auc > 0.78:
        print(f"\n~ Слабое улучшение над v2: AUC = {best_auc:.3f}")
        print(f"   Cross-compressor не даёт значимого прироста.")
    else:
        print(f"\n✗ Cross-compressor не помогает: AUC = {best_auc:.3f}")
        print(f"   v2 (0.82) остаётся лучшим результатом.")

    # ── Сохранение
    with open("multiscale_v3_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "snr", "auc", "mean_benign",
                    "mean_adv", "delta"])
        for r in snr_results:
            w.writerow([r["key"], f"{r['snr']:.4f}",
                        f"{r['auc']:.4f}", f"{r['mean_b']:.4f}",
                        f"{r['mean_a']:.4f}", f"{r['delta']:+.4f}"])
    print(f"\nСохранено: multiscale_v3_results.csv")

    # ── Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Top SNR
        ax = axes[0]
        top = sorted_by_snr[:15]
        names = [r["key"] for r in top][::-1]
        snrs = [r["snr"] for r in top][::-1]
        colors = ["crimson" if s > 0.5 else "steelblue" for s in snrs]
        ax.barh(range(len(names)), snrs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0.5, color="orange", linestyle="--",
                   label="SNR = 0.5")
        ax.axvline(1.0, color="green", linestyle="--",
                   label="SNR = 1.0")
        ax.set_xlabel("SNR")
        ax.set_title("Top-15 признаков по SNR")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        # LR
        ax = axes[1]
        names = [r["name"] for r in results_lr]
        aucs = [r["auc"] for r in results_lr]
        colors = ["crimson" if r["name"] == best_name
                  else "steelblue" for r in results_lr]
        bars = ax.barh(range(len(names)), aucs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.82, color="orange", linestyle="--",
                   alpha=0.5, label="v2 best (0.82)")
        ax.axvline(0.85, color="green", linestyle="--",
                   alpha=0.5, label="production")
        for b, v in zip(bars, aucs):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("AUC (5-fold)")
        ax.set_title("LR на разных наборах")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        # История
        ax = axes[2]
        history = ["v1\n(zlib)", "v2\n(zlib+bz2)", "v2\n(+lzma)",
                   f"v3\n({best_name[:15]})"]
        aucs_history = [0.64, 0.78, 0.82, best_auc]
        colors_h = ["gray", "steelblue", "steelblue", "crimson"]
        bars = ax.bar(history, aucs_history, color=colors_h)
        ax.axhline(0.85, color="green", linestyle="--",
                   label="production")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        for b, v in zip(bars, aucs_history):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=10)
        ax.set_ylabel("AUC")
        ax.set_title("История CIFAR-10")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig("multiscale_v3.png", dpi=120)
        print("Сохранено: multiscale_v3.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--compressors", type=str, nargs="+",
                    default=["zlib", "bz2"],
                    choices=["zlib", "bz2", "lzma"])
    args = ap.parse_args()

    run(n_samples=args.n, compressors=tuple(args.compressors))


if __name__ == "__main__":
    main()