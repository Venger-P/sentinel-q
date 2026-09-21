"""
multiscale_frag.py — мультимасштабный frag для CIFAR-10 (v2).

Исправление: frag считается как среднее по ансамблю компрессоров
(zlib + bz2 + lzma), а не только zlib.

В v1 был баг — только zlib, из-за чего SNR упал с 0.45 до 0.27
и результат был недооценён.

Масштабы: 32×32, 16×16, 8×8, 4×4 (через average pooling).
Признаки: frag, H, n_unique на каждом масштабе + дельты + профиль.

Запуск:
    python multiscale_frag.py --n 500
    python multiscale_frag.py --n 500 --compressors zlib bz2
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
    """
    img_np: (C, H, W) float [0,1].
    Возвращает байты downsample-изображения.

    factor=1 → 32×32 (оригинал)
    factor=2 → 16×16
    factor=4 → 8×8
    factor=8 → 4×4
    """
    if factor == 1:
        arr = (img_np * 255).astype(np.uint8)
        return arr.tobytes()

    t = torch.from_numpy(img_np).float()
    x = t.unsqueeze(0)  # (1, C, H, W)
    pooled = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    pooled_np = pooled.squeeze(0).numpy()
    arr = (pooled_np * 255).astype(np.uint8)
    return arr.tobytes()


def multiscale_features(img_np, compressors=("zlib", "bz2")):
    """
    img_np: (C, H, W) float [0,1].
    Возвращает словарь с признаками на всех масштабах.

    frag считается как среднее по ансамблю компрессоров.
    Дополнительно сохраняются отдельные значения по компрессорам.
    """
    out = {}
    frags_by_scale = {}

    for factor, name in [(1, "32"), (2, "16"), (4, "8"), (8, "4")]:
        data = image_to_bytes_at_scale(img_np, factor)

        # Мультикомпрессорный frag
        frags_list = [fragility(data, algo) for algo in compressors]
        f_avg = float(np.mean(frags_list))

        h = shannon_entropy(data)
        nu = n_unique_bytes(data)

        out[f"frag_{name}"] = f_avg
        for algo, f_val in zip(compressors, frags_list):
            out[f"frag_{algo[0]}_{name}"] = f_val  # frag_z_32, frag_b_32, ...

        out[f"H_{name}"] = h
        out[f"nu_{name}"] = nu
        frags_by_scale[name] = f_avg

    # Мультимасштабные дельты (профиль)
    out["d_frag_32_16"] = frags_by_scale["32"] - frags_by_scale["16"]
    out["d_frag_16_8"]  = frags_by_scale["16"] - frags_by_scale["8"]
    out["d_frag_8_4"]   = frags_by_scale["8"]  - frags_by_scale["4"]
    out["frag_mean"]    = float(np.mean(list(frags_by_scale.values())))
    out["frag_std"]     = float(np.std(list(frags_by_scale.values())))
    out["frag_range"]   = max(frags_by_scale.values()) - min(frags_by_scale.values())

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


# ── SNR и AUC для одного признака ───────────────────────────

def snr_for_feature(feature_b, feature_a):
    """SNR = |mean(a) - mean(b)| / std(b)."""
    delta = np.mean(feature_a) - np.mean(feature_b)
    std = np.std(feature_b)
    if std < 1e-9:
        return 0.0, delta
    return abs(delta) / std, delta


def auc_for_feature(feature_b, feature_a):
    """AUC через Mann-Whitney U."""
    n_b = len(feature_b)
    n_a = len(feature_a)
    all_vals = np.concatenate([feature_b, feature_a])
    order = np.argsort(all_vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(all_vals) + 1)
    rank_b = ranks[:n_b].sum()
    u = rank_b - n_b * (n_b + 1) / 2
    auc = u / (n_b * n_a)
    # Направление: adversarial обычно имеют меньший frag
    if np.mean(feature_a) < np.mean(feature_b):
        return auc
    return 1 - auc


# ── Основной эксперимент ────────────────────────────────────

def run(n_samples=500, compressors=("zlib", "bz2")):
    print("=" * 78)
    print("Multiscale frag для CIFAR-10 (v2, мультикомпрессорный)")
    print("=" * 78)
    print(f"Компрессоры: {compressors}")

    print(f"\n[1] Загрузка данных ...")
    x_b, y_b, x_a, y_a = load_cifar()
    n = min(n_samples, len(x_b), len(x_a))
    x_b = x_b[:n]
    x_a = x_a[:n]
    print(f"    {n} benign + {n} adversarial")
    print(f"    Форма: {x_b.shape}")

    # ── Признаки
    print(f"\n[2] Мультимасштабные признаки ...")
    t0 = time.time()

    feats_b = []
    for i in range(n):
        feats_b.append(multiscale_features(x_b[i], compressors))
        if (i + 1) % 100 == 0:
            print(f"    benign: {i+1}/{n}  ({time.time()-t0:.1f}s)")

    feats_a = []
    for i in range(n):
        feats_a.append(multiscale_features(x_a[i], compressors))
        if (i + 1) % 100 == 0:
            print(f"    adversarial: {i+1}/{n}  ({time.time()-t0:.1f}s)")

    keys = sorted(feats_b[0].keys())
    print(f"    Признаков на образец: {len(keys)}")
    print(f"    Всего: {len(feats_b)} × {len(keys)}")

    # ── SNR для каждого признака
    print(f"\n[3] SNR для каждого признака:")
    header = (f"    {'feature':<18} {'benign':>10} {'adversarial':>12} "
              f"{'Δ':>10} {'SNR':>8} {'AUC':>7}")
    print(header)
    print("    " + "-" * (len(header) - 4))

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
        print(f"    {key:<18} {fb.mean():>10.4f} {fa.mean():>12.4f} "
              f"{delta:>+10.4f} {snr:>8.3f} {auc:>7.3f}")

    # ── Baseline
    print(f"\n[4] Baseline (frag_32, мультикомпрессорный):")
    frag_32_b = np.array([f["frag_32"] for f in feats_b])
    frag_32_a = np.array([f["frag_32"] for f in feats_a])
    snr_base, _ = snr_for_feature(frag_32_b, frag_32_a)
    auc_base = auc_for_feature(frag_32_b, frag_32_a)
    print(f"    SNR = {snr_base:.3f}, AUC = {auc_base:.3f}")

    # ── Лучший одиночный признак
    best = max(snr_results, key=lambda r: r["snr"])
    print(f"\n[5] Лучший одиночный признак:")
    print(f"    {best['key']}: SNR = {best['snr']:.3f}, "
          f"AUC = {best['auc']:.3f}")

    # ── LR на разных наборах
    print(f"\n[6] Logistic Regression на разных наборах:")
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
        X = np.vstack([X_b, X_a])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        return X, y

    # Компрессорные ключи
    comp_keys_by_scale = []
    for s in ["32", "16", "8", "4"]:
        for algo in compressors:
            k = f"frag_{algo[0]}_{s}"
            if k in keys:
                comp_keys_by_scale.append(k)

    feature_sets = {
        "frag_32 (baseline)":     ["frag_32"],
        "all frags (avg)":        [f"frag_{s}" for s in ["32", "16", "8", "4"]],
        "frags by compressor":    comp_keys_by_scale,
        "frags + H":              ["frag_32", "frag_16", "frag_8", "frag_4",
                                    "H_32", "H_16", "H_8", "H_4"],
        "frags + deltas":         ["frag_32", "frag_16", "frag_8", "frag_4",
                                    "d_frag_32_16", "d_frag_16_8",
                                    "d_frag_8_4"],
        "frags + compressor deltas": comp_keys_by_scale + [
            "d_frag_32_16", "d_frag_16_8", "d_frag_8_4",
        ],
        "everything":             keys,
    }

    print(f"    {'features':<28} {'n_feat':>7} {'AUC (5-fold)':>16}")
    print("    " + "-" * 54)

    best_auc = 0
    best_name = None
    results_lr = []

    for name, keys_list in feature_sets.items():
        X, y = make_xy(keys_list)
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        auc = scores.mean()
        auc_std = scores.std()

        results_lr.append({
            "name": name, "auc": float(auc),
            "n_features": len(keys_list),
        })

        marker = ""
        if auc > best_auc:
            best_auc = auc
            best_name = name
            marker = " ←"

        print(f"    {name:<28} {len(keys_list):>7} "
              f"{auc:>10.4f} ± {auc_std:.4f}{marker}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    print(f"\nBaseline (frag_32):     SNR = {snr_base:.3f}, "
          f"AUC = {auc_base:.3f}")
    print(f"Best single:            {best['key']}, "
          f"SNR = {best['snr']:.3f}, AUC = {best['auc']:.3f}")
    print(f"Best LR:                {best_name}, "
          f"AUC = {best_auc:.3f}")

    improvement_snr = best["snr"] / max(snr_base, 1e-6)
    improvement_auc = best_auc - auc_base

    print(f"\nУлучшение SNR: ×{improvement_snr:.2f}")
    print(f"Улучшение AUC: {improvement_auc:+.3f}")

    # Сравнение с предыдущими экспериментами
    print(f"\nСравнение с предыдущими результатами:")
    print(f"  CIFAR v1 (только zlib, frag_32):   SNR = 0.27, AUC = 0.64")
    print(f"  CIFAR v2 (мультикомпрессор):       "
          f"SNR = {snr_base:.2f}, AUC = {auc_base:.2f}")
    print(f"  MNIST (baseline):                   SNR = 1.67, AUC = 0.99")
    print(f"  Текст (swap):                       SNR = 1.62, AUC = 0.997")

    # ── Вывод
    print(f"\n{'=' * 78}")
    if best_auc > 0.85:
        print(f"✓ Мультимасштаб + мультикомпрессор РАБОТАЕТ")
        print(f"   AUC = {best_auc:.3f} — Sentinel-Q применим к CIFAR-10.")
    elif best_auc > 0.80:
        print(f"~ Хороший результат: AUC = {best_auc:.3f}")
        print(f"   Значительно лучше baseline, приближается к production.")
    elif best_auc > 0.75:
        print(f"~ Умеренное улучшение: AUC = {best_auc:.3f}")
        print(f"   Детектор работает, но не production-ready.")
    elif best_auc > 0.68:
        print(f"~ Слабое улучшение: AUC = {best_auc:.3f}")
        print(f"   Мультимасштаб помогает, но не решает проблему.")
    else:
        print(f"✗ Мультимасштаб не помогает: AUC = {best_auc:.3f}")
        print(f"   Проблема CIFAR глубже, чем масштаб и компрессор.")

    # ── Сохранение
    with open("multiscale_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "snr", "auc", "mean_benign",
                    "mean_adv", "delta"])
        for r in snr_results:
            w.writerow([r["key"], f"{r['snr']:.4f}",
                        f"{r['auc']:.4f}", f"{r['mean_b']:.4f}",
                        f"{r['mean_a']:.4f}", f"{r['delta']:+.4f}"])
    print(f"\nСохранено: multiscale_results.csv")

    # ── Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # SNR по масштабам (усреднённый frag)
        ax = axes[0]
        scales = ["32", "16", "8", "4"]
        snr_by_scale = []
        for s in scales:
            key = f"frag_{s}"
            r = next((r for r in snr_results if r["key"] == key), None)
            snr_by_scale.append(r["snr"] if r else 0.0)

        colors = ["steelblue" if s > 0.5 else "crimson"
                  for s in snr_by_scale]
        bars = ax.bar(scales, snr_by_scale, color=colors)
        ax.axhline(1.0, color="green", linestyle="--",
                   label="SNR = 1 (порог применимости)")
        ax.axhline(0.5, color="orange", linestyle="--",
                   label="SNR = 0.5")
        for b, v in zip(bars, snr_by_scale):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xlabel("scale")
        ax.set_ylabel("SNR")
        ax.set_title("SNR по масштабам (мультикомпрессорный frag)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # AUC по масштабам
        ax = axes[1]
        auc_by_scale = []
        for s in scales:
            key = f"frag_{s}"
            r = next((r for r in snr_results if r["key"] == key), None)
            auc_by_scale.append(r["auc"] if r else 0.5)

        bars = ax.bar(scales, auc_by_scale, color="crimson")
        ax.axhline(0.5, color="gray", linestyle="--", label="random")
        ax.axhline(0.75, color="orange", linestyle="--",
                   label="baseline v1 (0.75)")
        for b, v in zip(bars, auc_by_scale):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xlabel("scale")
        ax.set_ylabel("AUC")
        ax.set_title("AUC по масштабам")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # LR результаты
        ax = axes[2]
        names = [r["name"] for r in results_lr]
        aucs = [r["auc"] for r in results_lr]
        colors_bar = ["crimson" if r["name"] == best_name
                      else "steelblue" for r in results_lr]
        bars = ax.barh(range(len(names)), aucs, color=colors_bar)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0.75, color="orange", linestyle="--",
                   alpha=0.5, label="baseline")
        ax.axvline(0.85, color="green", linestyle="--",
                   alpha=0.5, label="production")
        for b, v in zip(bars, aucs):
            ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("AUC (5-fold CV)")
        ax.set_title("LR на разных наборах признаков")
        ax.legend()
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig("multiscale_frag.png", dpi=120)
        print("Сохранено: multiscale_frag.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--compressors", type=str, nargs="+",
                    default=["zlib", "bz2"],
                    choices=["zlib", "bz2", "lzma"],
                    help="Ансамбль компрессоров (default: zlib bz2)")
    args = ap.parse_args()

    run(n_samples=args.n, compressors=tuple(args.compressors))


if __name__ == "__main__":
    main()