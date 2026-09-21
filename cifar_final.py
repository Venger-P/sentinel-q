"""
cifar_final.py — финальный Sentinel-Q v2.

Прорыв: hdiff_bit5 + hdiff_bit4 + td_bit* + frag_bit* дают AUC > 0.98.
Ключ — пространственные битовые признаки (SBF).

Задачи:
  1. Финальный greedy-набор на 1000+1000 (расширенный)
  2. Holdout проверка
  3. Confusion matrix и рабочие метрики
  4. Проверка на MNIST (не сломалось ли)
  5. Экспорт финальной модели

Запуск:
    python cifar_final.py --n 1000
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy, n_unique_bytes


# ── Финальный набор признаков ──────────────────────────────

def spatial_bit_features(img_np):
    """
    Пространственные битовые признаки (SBF).

    Для каждого битового слоя:
      - hdiff: средний горизонтальный градиент
      - vdiff: средний вертикальный градиент
      - td:    transition density
      - frag:  frag на байтовом представлении слоя
      - H:     энтропия
    """
    out = {}
    arr = (img_np * 255).astype(np.uint8)

    for bit in range(8):
        bp = ((arr >> bit) & 1).astype(np.uint8)
        bp_bytes = (bp * 255).tobytes()
        binary = bp.tobytes()

        out[f"frag_bit{bit}"] = fragility(bp_bytes, "zlib")
        out[f"H_bit{bit}"] = shannon_entropy(binary)

        flat = bp.flatten()
        if len(flat) > 1:
            out[f"td_bit{bit}"] = float(
                (flat[1:] != flat[:-1]).mean()
            )

        out[f"hdiff_bit{bit}"] = float(
            np.abs(bp[:, 1:] - bp[:, :-1]).mean()
        )
        out[f"vdiff_bit{bit}"] = float(
            np.abs(bp[1:, :] - bp[:-1, :]).mean()
        )

    # Базовое
    full = arr.tobytes()
    out["frag"] = fragility(full, "zlib")
    out["frag_b"] = fragility(full, "bz2")
    out["H"] = shannon_entropy(full)

    return out


# Greedy-набор из предыдущего эксперимента
GREEDY_FEATURES = [
    "hdiff_bit5", "hdiff_bit4", "td_bit5", "td_bit0",
    "frag_bit3", "td_bit6", "H_bit5", "frag_bit4",
    "td_bit4", "td_bit2",
]


def load_cifar():
    cache = Path("../Kvorb/V5.0/adv_cache_cifar")
    if not cache.exists():
        cache = Path("adv_cache_cifar")
    b = np.load(cache / "benign.npz")
    a = np.load(cache / "adversarial.npz")
    return b["x"], b["y"], a["x"], a["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--save", default="sentinel_q_cifar_v2.pkl")
    args = ap.parse_args()

    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                   precision_score, recall_score,
                                   f1_score, confusion_matrix)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (StratifiedKFold,
                                          cross_val_score,
                                          train_test_split)
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline

    print("=" * 80)
    print("Sentinel-Q v2 — финальная модель для CIFAR-10")
    print("=" * 80)

    x_b, y_b, x_a, y_a = load_cifar()
    n = min(args.n, len(x_b), len(x_a))
    x_b, x_a = x_b[:n], x_a[:n]
    print(f"\n{n} + {n} = {2*n} образцов")

    print(f"\n[1] Признаки ...")
    t0 = time.time()
    feats_b = [spatial_bit_features(x_b[i]) for i in range(n)]
    feats_a = [spatial_bit_features(x_a[i]) for i in range(n)]
    print(f"    {time.time()-t0:.1f}s")

    keys = sorted(feats_b[0].keys())
    X_b = np.array([[f[k] for k in keys] for f in feats_b])
    X_a = np.array([[f[k] for k in keys] for f in feats_a])
    X = np.vstack([X_b, X_a])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    # ── Сравнение наборов
    print(f"\n[2] Сравнение наборов:")

    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    sets = {
        "frag only (v1)":       ["frag", "frag_b", "H"],
        "frag + frag_bit":       ["frag", "frag_b", "H"] +
                                  [f"frag_bit{i}" for i in range(8)],
        "SBF greedy (10)":       GREEDY_FEATURES,
        "SBF greedy + frag":     GREEDY_FEATURES + ["frag", "frag_b"],
        "all 50 features":       keys,
    }

    results = []
    print(f"    {'set':<26} {'n':>4} {'CV AUC':>10} {'±':>8}")
    print("    " + "-" * 52)
    for name, ksub in sets.items():
        if not ksub or any(k not in keys for k in ksub):
            continue
        idx = [keys.index(k) for k in ksub]
        Xs = X[:, idx]
        scores = cross_val_score(pipe, Xs, y, cv=cv, scoring="roc_auc")
        results.append({"name": name, "auc": scores.mean(),
                        "std": scores.std(), "n": len(ksub),
                        "keys": ksub})
        print(f"    {name:<26} {len(ksub):>4} "
              f"{scores.mean():>10.4f} ±{scores.std():>7.4f}")

    # ── Holdout для финального набора
    print(f"\n[3] Holdout test (80/20) для SBF greedy (10):")
    final_keys = GREEDY_FEATURES
    idx = [keys.index(k) for k in final_keys]
    Xf = X[:, idx]
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xf, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe_h = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    pipe_h.fit(X_tr, y_tr)
    y_prob = pipe_h.predict_proba(X_te)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    auc_h = roc_auc_score(y_te, y_prob)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    print(f"    ROC-AUC:   {auc_h:.4f}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"\n    Confusion matrix:")
    print(f"                  pred_benign  pred_adv")
    print(f"      true_benign     {cm[0,0]:>6}     {cm[0,1]:>6}")
    print(f"      true_adv        {cm[1,0]:>6}     {cm[1,1]:>6}")

    # ── Экспорт модели
    print(f"\n[4] Экспорт модели:")
    full_pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    full_pipe.fit(Xf, y)

    model_data = {
        "keys": final_keys,
        "pipeline": full_pipe,
        "cv_auc_mean": results[2]["auc"],
        "cv_auc_std":  results[2]["std"],
        "holdout_auc": auc_h,
        "holdout_acc": acc,
        "n_train_samples": n,
    }
    with open(args.save, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Сохранено: {args.save}")

    # ── Анализ важности
    print(f"\n[5] Важность признаков (финальные коэффициенты):")
    lr = full_pipe.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    sorted_coefs = sorted(zip(final_keys, coefs),
                           key=lambda x: -abs(x[1]))
    for k, c in sorted_coefs:
        bar = "█" * int(abs(c) * 20)
        print(f"    {k:<18} {c:>+8.3f}  {bar}")

    # ── Финальная сводка
    print(f"\n{'=' * 80}")
    print("ИТОГ")
    print(f"{'=' * 80}")

    print(f"\n  Финальный набор: {len(final_keys)} признаков")
    print(f"  CV AUC (10-fold):  {results[2]['auc']:.4f} "
          f"± {results[2]['std']:.4f}")
    print(f"  Holdout AUC:       {auc_h:.4f}")
    print(f"  Holdout Accuracy:  {acc:.4f}")
    print(f"  Holdout F1:        {f1:.4f}")

    print(f"\n  История CIFAR-10:")
    print(f"    v1 (zlib only):        AUC = 0.640")
    print(f"    v3 (cross-comp):       AUC = 0.835")
    print(f"    v4 (log-transform):    AUC = 0.833")
    print(f"    v6 (all 50 features):  AUC = 0.99+")
    print(f"    v2 (SBF, 10 features): AUC = {auc_h:.3f}  ← ФИНАЛ")

    print(f"\n  Ключевые признаки:")
    print(f"    hdiff_bit5 — горизонтальный градиент 5-го бита")
    print(f"    hdiff_bit4 — горизонтальный градиент 4-го бита")
    print(f"    td_bit5    — transition density 5-го бита")
    print(f"    td_bit0    — transition density 0-го бита")

    if auc_h > 0.95:
        print(f"\n✓✓✓ PRODUCTION READY")
        print(f"    CIFAR-10: AUC = {auc_h:.4f} на holdout")
        print(f"    Модель сохранена в {args.save}")

    # ── Сохранение в CSV
    with open("cifar_final_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "n_features", "cv_auc", "cv_std"])
        for r in results:
            w.writerow([r["name"], r["n"],
                        f"{r['auc']:.4f}", f"{r['std']:.4f}"])
    print(f"\n  Сохранено: cifar_final_results.csv")


if __name__ == "__main__":
    main()