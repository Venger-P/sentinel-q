"""
cifar_exploration.py — поиск новых признаков для CIFAR-10.

Не frag на всём изображении, а:
  1. Per-channel frag (R, G, B отдельно)
  2. Y (luma) frag — там основная структура
  3. Bit-plane frag (8 битовых слоёв отдельно)
  4. 2D transpose frag (столбцы как байты)
  5. LZ76 algorithmic complexity
  6. Entropy rate H(x_i | x_{i-1})
  7. Mutual information соседних байтов
  8. Autocorrelation на разных лагах
  9. Singular values (Marchenko-Pastur)
  10. Run-length statistics

Запуск:
    python cifar_exploration.py --n 500
"""

import argparse
import sys
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


def lz76_complexity(data: bytes) -> float:
    """
    LZ76 сложность — нормализованная.

    Число различных подстрок в LZ-разложении. Не равно compression ratio.
    """
    n = len(data)
    if n == 0:
        return 0.0
    i, k, l = 0, 1, 1
    c = 1
    while True:
        if data[i:i+l] in data[:i+l-1] or i + l > n:
            c += 1
            i += l
            if i >= n:
                break
            l = 1
        else:
            l += 1
            if i + l > n:
                c += 1
                break
    return c * np.log2(n) / n


def entropy_rate(data: bytes, order: int = 1) -> float:
    """H(x_i | x_{i-1}, ..., x_{i-order})."""
    if len(data) <= order:
        return 0.0
    contexts = {}
    for i in range(order, len(data)):
        ctx = data[i - order:i]
        if ctx not in contexts:
            contexts[ctx] = Counter()
        contexts[ctx][data[i]] += 1
    total_entropy = 0.0
    total_count = 0
    for ctx_counts in contexts.values():
        s = sum(ctx_counts.values())
        total_entropy += entropy(ctx_counts) * s
        total_count += s
    return total_entropy / total_count if total_count > 0 else 0.0


def mutual_information_adjacent(data: bytes) -> float:
    """MI между соседними байтами: I(x_i; x_{i+1})."""
    if len(data) < 2:
        return 0.0
    xy = Counter(zip(data[:-1], data[1:]))
    x_counts = Counter(data[:-1])
    y_counts = Counter(data[1:])
    n = len(data) - 1
    mi = 0.0
    for (x, y), c in xy.items():
        p_xy = c / n
        p_x = x_counts[x] / n
        p_y = y_counts[y] / n
        mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-12) + 1e-12)
    return float(mi)


def autocorrelation(data: bytes, lag: int) -> float:
    """Автокорреляция байтов на заданном лаге."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
    if len(arr) <= lag:
        return 0.0
    a = arr[:-lag]
    b = arr[lag:]
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    if denom < 1e-9:
        return 0.0
    return float((a_c * b_c).sum() / denom)


def singular_value_ratio(data: bytes, shape=(32, 96)) -> dict:
    """Сингулярные значения байтовой матрицы."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        if len(arr) != shape[0] * shape[1]:
            return {"sv_ratio": 0.0, "sv_log_sum": 0.0}
        M = arr.reshape(shape)
        s = np.linalg.svd(M, compute_uv=False)
        s = s[s > 1e-9]
        if len(s) < 2:
            return {"sv_ratio": 0.0, "sv_log_sum": 0.0}
        return {
            "sv_ratio": float(s[0] / s[-1]),
            "sv_log_sum": float(np.log(s).sum()),
        }
    except Exception:
        return {"sv_ratio": 0.0, "sv_log_sum": 0.0}


def run_length_stats(data: bytes) -> dict:
    """Статистики длин последовательностей одинаковых байтов."""
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


# ── Полный набор признаков ──────────────────────────────────

def all_features(img_np):
    """
    img_np: (3, 32, 32) float [0,1].

    Возвращает словарь с ~40 признаками.
    """
    out = {}

    # Базовое представление
    full_bytes = (img_np * 255).astype(np.uint8).tobytes()
    out["frag_full"] = fragility(full_bytes, "zlib")
    out["frag_b_full"] = fragility(full_bytes, "bz2")
    out["H_full"] = shannon_entropy(full_bytes)
    out["nu_full"] = n_unique_bytes(full_bytes)

    # ── 1. Per-channel frag
    for ci, cname in enumerate(["R", "G", "B"]):
        ch = (img_np[ci] * 255).astype(np.uint8).tobytes()
        out[f"frag_{cname}"] = fragility(ch, "zlib")
        out[f"frag_b_{cname}"] = fragility(ch, "bz2")

    # ── 2. Y (luma) frag
    # Y = 0.299R + 0.587G + 0.114B
    y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
    y_bytes = (y * 255).astype(np.uint8).tobytes()
    out["frag_Y"] = fragility(y_bytes, "zlib")
    out["frag_b_Y"] = fragility(y_bytes, "bz2")
    out["H_Y"] = shannon_entropy(y_bytes)

    # ── 3. Bit-plane frag (8 слоёв)
    arr = (img_np * 255).astype(np.uint8)
    for bit in range(8):
        bp = ((arr >> bit) & 1).astype(np.uint8) * 255
        bp_bytes = bp.tobytes()
        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")
        out[f"H_bit{bit}"] = shannon_entropy(bp_bytes)

    # ── 4. Transpose frag
    arr_T = arr.transpose(0, 2, 1)
    out["frag_T"] = fragility(arr_T.tobytes(), "zlib")

    # ── 5. LZ76 complexity
    out["lz76_full"] = lz76_complexity(full_bytes)
    out["lz76_Y"] = lz76_complexity(y_bytes)

    # ── 6. Entropy rate
    out["er1_full"] = entropy_rate(full_bytes, order=1)
    out["er2_full"] = entropy_rate(full_bytes, order=2)
    out["er1_Y"] = entropy_rate(y_bytes, order=1)

    # ── 7. Mutual information
    out["mi_full"] = mutual_information_adjacent(full_bytes)
    out["mi_Y"] = mutual_information_adjacent(y_bytes)

    # ── 8. Autocorrelation
    for lag in [1, 2, 4, 8, 16]:
        out[f"ac{lag}_full"] = autocorrelation(full_bytes, lag)
        out[f"ac{lag}_Y"] = autocorrelation(y_bytes, lag)

    # ── 9. Singular values
    sv = singular_value_ratio(full_bytes, shape=(32, 96))
    out["sv_ratio"] = sv["sv_ratio"]
    out["sv_log_sum"] = sv["sv_log_sum"]

    # ── 10. Run-length
    rl = run_length_stats(full_bytes)
    out["rl_mean"] = rl["rl_mean"]
    out["rl_max"] = rl["rl_max"]
    out["rl_std"] = rl["rl_std"]

    return out


# ── Загрузка ────────────────────────────────────────────────

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
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print("=" * 80)
    print("CIFAR-10 — поиск новых признаков")
    print("=" * 80)

    x_b, y_b, x_a, y_a = load_cifar()
    n = min(args.n, len(x_b), len(x_a))
    x_b, x_a = x_b[:n], x_a[:n]
    print(f"\n{n} + {n}")

    print(f"\n[1] Признаки ...")
    import time
    t0 = time.time()
    feats_b = []
    for i in range(n):
        feats_b.append(all_features(x_b[i]))
        if (i + 1) % 100 == 0:
            print(f"    benign: {i+1}/{n}  ({time.time()-t0:.1f}s)")
    feats_a = []
    for i in range(n):
        feats_a.append(all_features(x_a[i]))
        if (i + 1) % 100 == 0:
            print(f"    adv:    {i+1}/{n}  ({time.time()-t0:.1f}s)")

    keys = sorted(feats_b[0].keys())
    X_b = np.array([[f[k] for k in keys] for f in feats_b])
    X_a = np.array([[f[k] for k in keys] for f in feats_a])
    X = np.vstack([X_b, X_a])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    print(f"\n    Признаков: {len(keys)}")

    # ── Univariate AUC
    print(f"\n[2] Univariate AUC (Top-25):")
    print(f"    {'feature':<18} {'AUC':>7} {'benign':>10} {'adv':>10}")
    print("    " + "-" * 48)

    univ = []
    for i, key in enumerate(keys):
        auc = roc_auc_score(y, X[:, i])
        if np.mean(X_a[:, i]) > np.mean(X_b[:, i]):
            auc = 1 - auc
        univ.append({"key": key, "auc": auc, "idx": i,
                     "mb": X_b[:, i].mean(), "ma": X_a[:, i].mean()})
    univ.sort(key=lambda r: -r["auc"])

    for r in univ[:25]:
        print(f"    {r['key']:<18} {r['auc']:>7.4f} "
              f"{r['mb']:>10.4f} {r['ma']:>10.4f}")

    # ── Baseline
    print(f"\n[3] LR на всём ({len(keys)} признаков):")
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print(f"    AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Только признаки с AUC > 0.65
    print(f"\n[4] LR на сильных признаках (AUC > 0.65):")
    sig = [r["key"] for r in univ if r["auc"] > 0.65]
    print(f"    {len(sig)} признаков: {', '.join(sig[:8])}...")
    if sig:
        idx = [keys.index(k) for k in sig]
        X_sig = X[:, idx]
        scores2 = cross_val_score(pipe, X_sig, y, cv=cv,
                                   scoring="roc_auc")
        print(f"    AUC = {scores2.mean():.4f} ± {scores2.std():.4f}")
    else:
        scores2 = None

    # ── Greedy forward selection
    print(f"\n[5] Greedy forward selection (10 шагов):")
    selected = []
    remaining = list(range(len(keys)))
    current_auc = 0.5
    for step in range(10):
        best_idx, best_auc = None, current_auc
        for idx in remaining:
            trial = selected + [idx]
            X_sub = X[:, trial]
            auc = cross_val_score(pipe, X_sub, y, cv=cv,
                                   scoring="roc_auc").mean()
            if auc > best_auc:
                best_auc = auc
                best_idx = idx
        if best_idx is None:
            print(f"    Шаг {step+1}: нет улучшения. Стоп.")
            break
        gain = best_auc - current_auc
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_auc = best_auc
        print(f"    Шаг {step+1:>2}: + {keys[best_idx]:<18} "
              f"AUC = {current_auc:.4f}  (+{gain:.4f})")

    # ── Итог
    print(f"\n{'=' * 80}")
    print("ИТОГ")
    print(f"{'=' * 80}")
    print(f"\n  Baseline (все {len(keys)}):     AUC = {scores.mean():.4f}")
    if scores2 is not None:
        print(f"  Сильные ({len(sig)}):            AUC = {scores2.mean():.4f}")
    print(f"  Greedy ({len(selected)}):            AUC = {current_auc:.4f}")

    print(f"\n  История CIFAR-10:")
    print(f"    v1 (zlib):                     AUC = 0.640")
    print(f"    v3 (cross-comp):               AUC = 0.835  ← было лучшим")
    print(f"    exploration (new features):    AUC = {current_auc:.4f}")

    if current_auc > 0.95:
        print(f"\n✓✓✓ ПРОРЫВ: AUC = {current_auc:.4f}")
        print(f"    CIFAR-10 решён новыми признаками!")
    elif current_auc > 0.88:
        print(f"\n✓✓ ОТЛИЧНО: AUC = {current_auc:.4f}")
    elif current_auc > 0.835:
        print(f"\n✓ Улучшение над v3: AUC = {current_auc:.4f}")
    else:
        print(f"\n✗ Не улучшили v3: AUC = {current_auc:.4f}")

    # ── Сохранение
    import csv
    with open("cifar_exploration.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "auc",
                    "mean_benign", "mean_adv"])
        for i, r in enumerate(univ):
            w.writerow([i+1, r["key"], f"{r['auc']:.4f}",
                        f"{r['mb']:.4f}", f"{r['ma']:.4f}"])
    print(f"\nСохранено: cifar_exploration.csv")


if __name__ == "__main__":
    main()