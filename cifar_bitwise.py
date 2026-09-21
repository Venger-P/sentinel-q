"""
cifar_bitwise.py — финальный скрипт: bit-level Sentinel-Q.

Находка: frag + bit даёт AUC 0.93 (было 0.75 на frag only).
Ключ — не compression, а битовые слои. Adversarial-шум живёт в
младших битах, старшие сохраняют структуру.

Задачи:
  1. Найти top-N битовых признаков
  2. Компактный набор (5-7 признаков)
  3. Holdout test
  4. Финальная сводка

Запуск:
    python cifar_bitwise.py --n 500
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy, n_unique_bytes


# ── Битовые признаки ────────────────────────────────────────

def bit_plane_features(img_np):
    """
    Разбивает изображение на 8 битовых слоёв.
    Для каждого слоя — frag, H, nu, transition density.
    """
    out = {}
    arr = (img_np * 255).astype(np.uint8)

    for bit in range(8):
        bp = ((arr >> bit) & 1).astype(np.uint8)
        bp_bytes = (bp * 255).tobytes()
        binary = bp.tobytes()  # 0/1 байты

        # frag на байтах (0 или 255)
        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")
        # H на битовом слое (энтропия 0/1)
        out[f"H_bit{bit}"] = shannon_entropy(binary)
        # n_unique
        out[f"nu_bit{bit}"] = n_unique_bytes(binary)

        # Transition density: доля смен 0→1 или 1→0
        flat = bp.flatten()
        if len(flat) > 1:
            transitions = (flat[1:] != flat[:-1]).mean()
            out[f"td_bit{bit}"] = float(transitions)

        # Изменение относительно соседей (2D-градиент)
        # Считаем разницу по горизонтали и вертикали
        h_diff = np.abs(bp[:, 1:] - bp[:, :-1]).mean()
        v_diff = np.abs(bp[1:, :] - bp[:-1, :]).mean()
        out[f"hdiff_bit{bit}"] = float(h_diff)
        out[f"vdiff_bit{bit}"] = float(v_diff)

    # Базовые признаки (для сравнения)
    full_bytes = arr.tobytes()
    out["frag"] = fragility(full_bytes, "zlib")
    out["frag_b"] = fragility(full_bytes, "bz2")
    out["H"] = shannon_entropy(full_bytes)
    out["nu"] = n_unique_bytes(full_bytes)

    # Y-канал
    y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
    y_bytes = (y * 255).astype(np.uint8).tobytes()
    out["frag_Y"] = fragility(y_bytes, "zlib")
    out["H_Y"] = shannon_entropy(y_bytes)

    return out


def load_cifar():
    cache = Path("../Kvorb/V5.0/adv_cache_cifar")
    if not cache.exists():
        cache = Path("adv_cache_cifar")
    b = np.load(cache / "benign.npz")
    a = np.load(cache / "adversarial.npz")
    return b["x"], b["y"], a["x"], a["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (StratifiedKFold,
                                          cross_val_score,
                                          train_test_split)
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline

    print("=" * 80)
    print("CIFAR-10 — bit-level Sentinel-Q")
    print("=" * 80)

    x_b, y_b, x_a, y_a = load_cifar()
    n = min(args.n, len(x_b), len(x_a))
    x_b, x_a = x_b[:n], x_a[:n]
    print(f"\n{n} + {n}")

    print(f"\n[1] Признаки ...")
    t0 = time.time()
    feats_b = [bit_plane_features(x_b[i]) for i in range(n)]
    feats_a = [bit_plane_features(x_a[i]) for i in range(n)]
    print(f"    {time.time()-t0:.1f}s")

    keys = sorted(feats_b[0].keys())
    X_b = np.array([[f[k] for k in keys] for f in feats_b])
    X_a = np.array([[f[k] for k in keys] for f in feats_a])
    X = np.vstack([X_b, X_a])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"    Признаков: {len(keys)}")

    # ── Univariate
    print(f"\n[2] Univariate AUC (Top-15):")
    print(f"    {'feature':<20} {'AUC':>7}")
    print("    " + "-" * 30)
    univ = []
    for i, key in enumerate(keys):
        auc = roc_auc_score(y, X[:, i])
        if X_a[:, i].mean() > X_b[:, i].mean():
            auc = 1 - auc
        univ.append({"key": key, "auc": auc, "idx": i})
    univ.sort(key=lambda r: -r["auc"])
    for r in univ[:15]:
        print(f"    {r['key']:<20} {r['auc']:>7.4f}")

    # ── Компактные наборы
    print(f"\n[3] Компактные наборы:")

    bit_keys = [k for k in keys if "bit" in k]
    frag_bit_keys = [k for k in keys if k.startswith("frag_bit")]
    H_bit_keys = [k for k in keys if k.startswith("H_bit")]
    td_bit_keys = [k for k in keys if k.startswith("td_bit")]
    hdiff_keys = [k for k in keys if k.startswith("hdiff_bit")]
    vdiff_keys = [k for k in keys if k.startswith("vdiff_bit")]

    # Top-битовые признаки по univariate
    top_bit = [r["key"] for r in univ if "bit" in r["key"]][:6]

    sets = {
        "frag only (baseline)":       ["frag", "frag_b", "H", "nu"],
        "frag + all bits":            ["frag", "frag_b"] + bit_keys,
        "top-6 bit features":          top_bit,
        "frag_bit all 8":             frag_bit_keys,
        "frag_bit + H_bit":           frag_bit_keys + H_bit_keys,
        "frag + frag_bit":            ["frag", "frag_b"] + frag_bit_keys,
        "frag + frag_bit + H_bit":    ["frag", "frag_b"]
                                       + frag_bit_keys + H_bit_keys,
        "frag + td_bit":              ["frag", "frag_b"] + td_bit_keys,
        "everything":                 keys,
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )

    results = []
    print(f"    {'set':<32} {'n':>4} {'AUC':>10} {'±':>8}")
    print("    " + "-" * 58)
    for name, ksub in sets.items():
        if not ksub:
            continue
        idx = [keys.index(k) for k in ksub]
        Xs = X[:, idx]
        scores = cross_val_score(pipe, Xs, y, cv=cv, scoring="roc_auc")
        results.append({"name": name, "auc": scores.mean(),
                        "std": scores.std(), "n": len(ksub)})
        print(f"    {name:<32} {len(ksub):>4} "
              f"{scores.mean():>10.4f} ±{scores.std():>7.4f}")

    # ── Holdout
    print(f"\n[4] Holdout test (80/20):")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, ksub in [("frag only", ["frag", "frag_b", "H", "nu"]),
                        ("top-6 bits", top_bit),
                        ("frag + frag_bit",
                         ["frag", "frag_b"] + frag_bit_keys),
                        ("frag + frag_bit + H_bit",
                         ["frag", "frag_b"] + frag_bit_keys + H_bit_keys)]:
        idx = [keys.index(k) for k in ksub]
        Xtr = X_tr[:, idx]
        Xte = X_te[:, idx]
        pipe_h = make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        pipe_h.fit(Xtr, y_tr)
        auc_h = roc_auc_score(y_te, pipe_h.predict_proba(Xte)[:, 1])
        print(f"    {name:<28} holdout AUC = {auc_h:.4f}")

    # ── Greedy
    print(f"\n[5] Greedy forward (10 шагов):")
    selected = []
    remaining = list(range(len(keys)))
    current_auc = 0.5
    for step in range(10):
        best_idx, best_auc = None, current_auc
        for idx in remaining:
            trial = selected + [idx]
            Xs = X[:, trial]
            auc = cross_val_score(pipe, Xs, y, cv=cv,
                                   scoring="roc_auc").mean()
            if auc > best_auc:
                best_auc = auc
                best_idx = idx
        if best_idx is None:
            break
        gain = best_auc - current_auc
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_auc = best_auc
        print(f"    Шаг {step+1:>2}: + {keys[best_idx]:<20} "
              f"AUC = {current_auc:.4f}  (+{gain:.4f})")

    # ── Финальная сводка
    print(f"\n{'=' * 80}")
    print("ИТОГ")
    print(f"{'=' * 80}")

    best = max(results, key=lambda r: r["auc"])
    print(f"\n  Лучший набор: {best['name']}")
    print(f"    Признаков: {best['n']}")
    print(f"    CV AUC:    {best['auc']:.4f} ± {best['std']:.4f}")

    print(f"\n  Greedy ({len(selected)}): AUC = {current_auc:.4f}")
    print(f"    Набор: {', '.join([keys[i] for i in selected])}")

    print(f"\n  История CIFAR-10:")
    print(f"    v1 (zlib only):           AUC = 0.640")
    print(f"    v3 (cross-comp):          AUC = 0.835")
    print(f"    exploration (52 feat):    AUC = 0.9898")
    print(f"    bitwise (compact):        AUC = {best['auc']:.4f}")

    if best["auc"] > 0.95 and best["n"] <= 15:
        print(f"\n✓✓✓ ПРОРЫВ ПОДТВЕРЖДЁН")
        print(f"    Компактный набор ({best['n']} признаков) "
              f"даёт production AUC = {best['auc']:.4f}")
    elif best["auc"] > 0.95:
        print(f"\n✓✓ Работает, но набор велик ({best['n']})")
    else:
        print(f"\n~ Умеренный результат")

    # Сохранение
    with open("cifar_bitwise.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "univariate_auc"])
        for i, r in enumerate(univ):
            w.writerow([i+1, r["key"], f"{r['auc']:.4f}"])
    print(f"\nСохранено: cifar_bitwise.csv")


if __name__ == "__main__":
    main()