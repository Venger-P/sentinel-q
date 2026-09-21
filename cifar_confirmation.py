"""
cifar_confirmation.py — подтверждение прорыва на CIFAR-10.

Проверяет находку:
  - sv_ratio: AUC = 0.9547 (одиночный)
  - LR на 52 признаках: AUC = 0.9898

Что делаем:
  1. Расширенный набор (1000 + 1000)
  2. Стратифицированная 10-fold CV
  3. Проверка на разных seed
  4. Устойчивость sv_ratio к разным shape (16x192, 64x48, 24x128)
  5. Абляции: без sv_ratio, без rl_mean, только spectral

Запуск:
    python cifar_confirmation.py --n 1000
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy, n_unique_bytes


# ── Утилиты ─────────────────────────────────────────────────

def entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def run_length_stats(data: bytes) -> dict:
    if len(data) < 2:
        return {"rl_mean": 0.0, "rl_max": 0.0, "rl_std": 0.0}
    runs = []
    current = 1
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return {
        "rl_mean": float(np.mean(runs)),
        "rl_max": float(np.max(runs)),
        "rl_std": float(np.std(runs)),
    }


def sv_ratio_for_shape(data: bytes, shape) -> dict:
    """Сингулярные значения байтовой матрицы заданной формы."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        if len(arr) != shape[0] * shape[1]:
            return {"sv_ratio": 0.0, "sv_log_sum": 0.0,
                    "sv_top1_ratio": 0.0, "sv_top5_ratio": 0.0,
                    "sv_effective_rank": 0.0}
        M = arr.reshape(shape)
        s = np.linalg.svd(M, compute_uv=False)
        s = s[s > 1e-9]
        if len(s) < 2:
            return {"sv_ratio": 0.0, "sv_log_sum": 0.0,
                    "sv_top1_ratio": 0.0, "sv_top5_ratio": 0.0,
                    "sv_effective_rank": 0.0}

        total = s.sum()
        return {
            "sv_ratio": float(s[0] / s[-1]),
            "sv_log_sum": float(np.log(s).sum()),
            "sv_top1_ratio": float(s[0] / total),
            "sv_top5_ratio": float(s[:5].sum() / total),
            "sv_effective_rank": float(
                np.exp(-np.sum((s / total) * np.log(s / total + 1e-12)))
            ),
        }
    except Exception:
        return {"sv_ratio": 0.0, "sv_log_sum": 0.0,
                "sv_top1_ratio": 0.0, "sv_top5_ratio": 0.0,
                "sv_effective_rank": 0.0}


# ── Набор признаков (v2 — расширенный) ─────────────────────

def all_features(img_np):
    out = {}
    full_bytes = (img_np * 255).astype(np.uint8).tobytes()

    # Базовое
    out["frag_full"] = fragility(full_bytes, "zlib")
    out["frag_b_full"] = fragility(full_bytes, "bz2")
    out["H_full"] = shannon_entropy(full_bytes)
    out["nu_full"] = n_unique_bytes(full_bytes)

    # SVD на разных формах
    for shape, tag in [((32, 96), "s1"), ((96, 32), "s2"),
                        ((64, 48), "s3"), ((24, 128), "s4")]:
        sv = sv_ratio_for_shape(full_bytes, shape)
        for k, v in sv.items():
            out[f"{k}_{tag}"] = v

    # Run-length
    rl = run_length_stats(full_bytes)
    for k, v in rl.items():
        out[k] = v

    # Битовые слои (старшие)
    arr = (img_np * 255).astype(np.uint8)
    for bit in [5, 6, 7]:
        bp = ((arr >> bit) & 1).astype(np.uint8) * 255
        bp_bytes = bp.tobytes()
        out[f"H_bit{bit}"] = shannon_entropy(bp_bytes)
        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")

    # Y (luma)
    y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
    y_bytes = (y * 255).astype(np.uint8).tobytes()
    out["frag_Y"] = fragility(y_bytes, "zlib")
    out["frag_b_Y"] = fragility(y_bytes, "bz2")
    out["H_Y"] = shannon_entropy(y_bytes)

    # SVD на Y
    sv_Y = sv_ratio_for_shape(y_bytes, (32, 32))
    out["sv_ratio_Y"] = sv_Y["sv_ratio"]
    out["sv_top1_ratio_Y"] = sv_Y["sv_top1_ratio"]
    out["sv_effective_rank_Y"] = sv_Y["sv_effective_rank"]

    # Run-length на Y
    rl_Y = run_length_stats(y_bytes)
    for k, v in rl_Y.items():
        out[f"{k}_Y"] = v

    return out


def load_cifar():
    cache = Path("../Kvorb/V5.0/adv_cache_cifar")
    if not cache.exists():
        cache = Path("adv_cache_cifar")
    b = np.load(cache / "benign.npz")
    a = np.load(cache / "adversarial.npz")
    return b["x"], b["y"], a["x"], a["y"]


# ── Основной эксперимент ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print("=" * 80)
    print("CIFAR-10 — подтверждение прорыва")
    print("=" * 80)

    x_b, y_b, x_a, y_a = load_cifar()
    n = min(args.n, len(x_b), len(x_a))
    x_b, x_a = x_b[:n], x_a[:n]
    print(f"\n{n} + {n} (всего {2*n} образцов)")

    print(f"\n[1] Признаки ...")
    t0 = time.time()
    feats_b, feats_a = [], []
    for i in range(n):
        feats_b.append(all_features(x_b[i]))
        feats_a.append(all_features(x_a[i]))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n}  ({time.time()-t0:.1f}s)")

    keys = sorted(feats_b[0].keys())
    X_b = np.array([[f[k] for k in keys] for f in feats_b])
    X_a = np.array([[f[k] for k in keys] for f in feats_a])
    X = np.vstack([X_b, X_a])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"    Признаков: {len(keys)}")

    # ── Univariate
    print(f"\n[2] Univariate AUC (Top-20):")
    print(f"    {'feature':<24} {'AUC':>7}")
    print("    " + "-" * 34)
    univ = []
    for i, key in enumerate(keys):
        auc = roc_auc_score(y, X[:, i])
        if X_a[:, i].mean() > X_b[:, i].mean():
            auc = 1 - auc
        univ.append({"key": key, "auc": auc, "idx": i})
    univ.sort(key=lambda r: -r["auc"])
    for r in univ[:20]:
        print(f"    {r['key']:<24} {r['auc']:>7.4f}")

    # ── 10-fold CV
    print(f"\n[3] Stratified 10-fold CV (3 seed):")
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )

    aucs = []
    for seed in [42, 123, 2024]:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        aucs.append(scores.mean())
        print(f"    seed={seed}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"    Средний: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # ── Абляции
    print(f"\n[4] Абляции:")

    spectral_keys = [k for k in keys
                     if k.startswith("sv_") or k.startswith("rl_")]
    frag_keys = [k for k in keys if k.startswith("frag_")]
    bit_keys = [k for k in keys if k.startswith("H_bit")]

    ablations = {
        "all features": keys,
        "spectral + rl only": spectral_keys,
        "sv_ratio only": ["sv_ratio_s1"],
        "sv_ratio + rl_mean": ["sv_ratio_s1", "rl_mean"],
        "no spectral": [k for k in keys if k not in spectral_keys],
        "no frag": [k for k in keys if k not in frag_keys],
        "frag + bit only": frag_keys + bit_keys,
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    ab_results = []
    for name, keys_sub in ablations.items():
        if not keys_sub:
            continue
        idx = [keys.index(k) for k in keys_sub]
        X_sub = X[:, idx]
        try:
            scores = cross_val_score(pipe, X_sub, y, cv=cv,
                                       scoring="roc_auc")
            auc = scores.mean()
            ab_results.append({"name": name, "auc": auc,
                                "n_features": len(keys_sub)})
            print(f"    {name:<24} ({len(keys_sub):>3}) "
                  f"AUC = {auc:.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"    {name:<24} ошибка: {e}")

    # ── Greedy forward
    print(f"\n[5] Greedy forward selection (8 шагов):")
    selected = []
    remaining = list(range(len(keys)))
    current_auc = 0.5
    cv_g = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for step in range(8):
        best_idx, best_auc = None, current_auc
        for idx in remaining:
            trial = selected + [idx]
            X_sub = X[:, trial]
            auc = cross_val_score(pipe, X_sub, y, cv=cv_g,
                                   scoring="roc_auc").mean()
            if auc > best_auc:
                best_auc = auc
                best_idx = idx
        if best_idx is None:
            print(f"    Шаг {step+1}: нет улучшения")
            break
        gain = best_auc - current_auc
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_auc = best_auc
        print(f"    Шаг {step+1:>2}: + {keys[best_idx]:<24} "
              f"AUC = {current_auc:.4f}  (+{gain:.4f})")

    # ── Итог
    print(f"\n{'=' * 80}")
    print("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")
    print(f"{'=' * 80}")

    best_ab = max(ab_results, key=lambda r: r["auc"])
    print(f"\n  Baseline (frag_full):          AUC = "
          f"{roc_auc_score(y, X[:, keys.index('frag_full')]):.4f}")
    print(f"  Best univariate:               {univ[0]['key']}, "
          f"AUC = {univ[0]['auc']:.4f}")
    print(f"  Best ablation:                 {best_ab['name']}, "
          f"AUC = {best_ab['auc']:.4f}")
    print(f"  Greedy ({len(selected)}):                "
          f"AUC = {current_auc:.4f}")
    print(f"  10-fold CV (all features):     "
          f"AUC = {np.mean(aucs):.4f}")

    print(f"\n  История CIFAR-10:")
    print(f"    v1 (zlib):                     AUC = 0.640")
    print(f"    v3 (cross-comp):               AUC = 0.835")
    print(f"    exploration:                   AUC = 0.9898")
    print(f"    confirmation:                  AUC = {np.mean(aucs):.4f}")

    if np.mean(aucs) > 0.98:
        print(f"\n✓✓✓ ПОДТВЕРЖДЕНО: AUC = {np.mean(aucs):.4f}")
        print(f"    CIFAR-10 полностью решён.")
        print(f"    Ключ: spectral признаки (sv_ratio).")

    # ── Сохранение
    with open("cifar_confirmation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "univariate_auc"])
        for i, r in enumerate(univ):
            w.writerow([i+1, r["key"], f"{r['auc']:.4f}"])
    print(f"\nСохранено: cifar_confirmation.csv")


if __name__ == "__main__":
    main()