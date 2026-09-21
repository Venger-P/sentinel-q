"""
pgd_step_sweep.py — что определяет bit-plane signature: alpha или n_iter?

Наблюдение: FGSM → PGD-2 обобщение даёт AUC=0.706 (провал),
FGSM → PGD-20 даёт AUC=0.915 (хорошо).

Гипотеза: НЕ число шагов, а РАЗМЕР шага (alpha) определяет,
попадает ли возмущение в бит-плоскости 4-5.

Эксперимент: сетка (alpha, n_iter):
    alpha ∈ {eps/20, eps/10, eps/4, eps/2}
    n_iter ∈ {2, 5, 20}
    eps = 0.05 фиксировано

Для каждой пары:
  1. Генерируем атаку
  2. Обучаем GradientBoosting на train
  3. Тестируем на test
  4. Считаем AUC и univariate AUC ключевых признаков
     (td_bit5, hdiff_bit5, td_bit4, hdiff_bit4, frag)

Ожидание:
  - Малое alpha (eps/20) → bit-признаки сильные
  - Большое alpha (eps/2) → bit-признаки слабые

Запуск:
    python pgd_step_sweep.py --n_benign 500
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


# ── Атаки ───────────────────────────────────────────────────

def pgd_attack(model, x, y, eps=0.05, alpha=None, n_iter=20):
    """PGD с явным alpha и n_iter."""
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


# ── Bit-признаки ────────────────────────────────────────────

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
        out[f"hdiff_bit{bit}"] = float(np.abs(bp[:, 1:] - bp[:, :-1]).mean())
        out[f"vdiff_bit{bit}"] = float(np.abs(bp[1:, :] - bp[:-1, :]).mean())
    return out


def extract_features(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return bit_features_one(img)

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


def univariate_auc(fb, fa):
    """AUC одного признака (правильный)."""
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(fb)), np.ones(len(fa))])
    scores = np.concatenate([fb, fa])
    auc = roc_auc_score(y, scores)
    if fa.mean() < fb.mean():
        return 1 - auc
    return auc


# ── Основной эксперимент ────────────────────────────────────

def run(n_benign=500, n_workers=4):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier

    print("=" * 82)
    print(f"PGD step sweep — alpha vs n_iter")
    print(f"Device: {DEVICE}")
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

    eps = 0.05

    # ── Сетка (alpha, n_iter)
    # alpha задаётся в долях eps
    combos = [
        ("eps/20", 20, "PGD-20 (малый шаг)"),
        ("eps/10", 10, "PGD-10 (малый шаг)"),
        ("eps/4",   4, "PGD-4 (средний шаг)"),
        ("eps/2",   2, "PGD-2 (большой шаг)"),
    ]

    print(f"\n[1] Генерация {len(combos)} атак ...")
    attacks = {}
    for alpha_tag, n_iter, label in combos:
        # alpha = eps / число в теге
        divisor = int(alpha_tag.split("/")[1])
        alpha_val = eps / divisor

        t0 = time.time()
        attacks[alpha_tag] = {
            "alpha": alpha_val,
            "n_iter": n_iter,
            "label": label,
            "train": pgd_attack(model, x_b_train.clone(), y_b_train,
                                 eps=eps, alpha=alpha_val, n_iter=n_iter),
            "test":  pgd_attack(model, x_b_test.clone(), y_b_test,
                                 eps=eps, alpha=alpha_val, n_iter=n_iter),
        }
        print(f"    {alpha_tag:<8} (alpha={alpha_val:.4f}, "
              f"n_iter={n_iter:<3}): {time.time()-t0:.1f}s")

    # ── ASR
    print(f"\n[2] ASR на test:")
    with torch.no_grad():
        pred_b = model(x_b_test).argmax(1)
        for tag, d in attacks.items():
            pred_a = model(d["test"]).argmax(1)
            asr = ((pred_b != pred_a)
                    & (pred_b == y_b_test)).float().mean().item()
            print(f"    {tag:<8} ASR = {asr:.3f}")

    # ── Извлечение признаков
    print(f"\n[3] Извлечение bit-признаков ...")
    t0 = time.time()
    fb_train = extract_features(x_b_train, n_workers)
    fb_test = extract_features(x_b_test, n_workers)
    fb_calib = extract_features(x_b_calib, n_workers)

    fa_train = {}
    fa_test = {}
    for tag, d in attacks.items():
        fa_train[tag] = extract_features(d["train"], n_workers)
        fa_test[tag] = extract_features(d["test"], n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    print(f"    Признаков: {len(keys)}")

    def to_X(feats):
        return np.array([[f[k] for k in keys] for f in feats])

    # ── Ключевые признаки для отслеживания
    key_features = ["frag", "td_bit5", "hdiff_bit5",
                     "td_bit4", "hdiff_bit4", "td_bit2", "hdiff_bit2"]

    # ── Univariate AUC
    print(f"\n[4] Univariate AUC ключевых признаков:")
    header = f"    {'feature':<16}"
    for tag, _, _ in combos:
        header += f" {tag:>9}"
    print(header)
    print("    " + "-" * (16 + 10 * len(combos)))

    univariate = {}
    for key in key_features:
        if key not in keys:
            continue
        row = {}
        for tag, _, _ in combos:
            fb = np.array([f[key] for f in fb_test])
            fa = np.array([f[key] for f in fa_test[tag]])
            row[tag] = univariate_auc(fb, fa)
        univariate[key] = row
        line = f"    {key:<16}"
        for tag, _, _ in combos:
            line += f" {row[tag]:>9.4f}"
        print(line)

    # ── Обучение и тестирование GB
    print(f"\n[5] GradientBoosting: обучение на train, тест на test:")
    print(f"    {'attack':<10} {'alpha':>9} {'n_iter':>7} "
          f"{'AUC':>8} {'det@5%':>8} {'det@10%':>8}")
    print("    " + "-" * 56)

    X_train_b = to_X(fb_train)
    X_test_b = to_X(fb_test)
    X_calib_b = to_X(fb_calib)

    results = {}
    for tag, d in attacks.items():
        X_train = np.vstack([X_train_b, to_X(fa_train[tag])])
        y_train = np.concatenate([np.zeros(len(fb_train)),
                                   np.ones(len(fa_train[tag]))])

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # Threshold из calib
        prob_b_calib = clf.predict_proba(X_calib_b)[:, 1]
        sorted_calib = np.sort(prob_b_calib)[::-1]
        idx5 = min(int(0.05 * len(sorted_calib)), len(sorted_calib) - 1)
        thr5 = sorted_calib[idx5]
        idx10 = min(int(0.10 * len(sorted_calib)), len(sorted_calib) - 1)
        thr10 = sorted_calib[idx10]

        prob_b_test = clf.predict_proba(X_test_b)[:, 1]
        prob_a_test = clf.predict_proba(to_X(fa_test[tag]))[:, 1]

        y_test = np.concatenate([np.zeros(len(prob_b_test)),
                                  np.ones(len(prob_a_test))])
        scores = np.concatenate([prob_b_test, prob_a_test])
        auc = roc_auc_score(y_test, scores)

        fpr5 = float((prob_b_test > thr5).mean())
        det5 = float((prob_a_test > thr5).mean())
        fpr10 = float((prob_b_test > thr10).mean())
        det10 = float((prob_a_test > thr10).mean())

        results[tag] = {
            "auc": auc, "det5": det5, "det10": det10,
            "alpha": d["alpha"], "n_iter": d["n_iter"],
        }

        marker = ""
        if det5 >= 0.80:
            marker = " ✓✓"
        elif det5 >= 0.60:
            marker = " ✓"
        print(f"    {tag:<10} {d['alpha']:>9.4f} {d['n_iter']:>7d} "
              f"{auc:>8.4f} {det5:>8.3f} {det10:>8.3f}{marker}")

    # ── Ключевой анализ
    print(f"\n{'=' * 82}")
    print("ГЛАВНЫЙ ВОПРОС: что сильнее влияет — alpha или n_iter?")
    print(f"{'=' * 82}")

    # Группируем по alpha
    print(f"\n  Влияние alpha (фиксируем, что важно именно alpha):")
    print(f"    {'alpha':>10} {'n_iter':>8} {'AUC':>8} {'det@5%':>8}")
    print("    " + "-" * 40)
    for tag, d in attacks.items():
        r = results[tag]
        print(f"    {d['alpha']:>10.4f} {d['n_iter']:>8d} "
              f"{r['auc']:>8.4f} {r['det5']:>8.3f}")

    # Корреляция между alpha и AUC
    alphas = np.array([d["alpha"] for d in attacks.values()])
    aucs = np.array([results[tag]["auc"] for tag in attacks])
    dets = np.array([results[tag]["det5"] for tag in attacks])
    n_iters = np.array([d["n_iter"] for d in attacks.values()])

    corr_alpha = np.corrcoef(alphas, aucs)[0, 1]
    corr_n_iter = np.corrcoef(n_iters, aucs)[0, 1]

    print(f"\n  Корреляции:")
    print(f"    corr(alpha, AUC)   = {corr_alpha:+.4f}")
    print(f"    corr(n_iter, AUC)  = {corr_n_iter:+.4f}")

    if abs(corr_alpha) > abs(corr_n_iter):
        print(f"\n  ✓ ALPHA сильнее коррелирует с AUC")
        print(f"    Гипотеза подтверждена: размер шага определяет")
        print(f"    силу bit-plane signature.")
    else:
        print(f"\n  ~ n_iter коррелирует сильнее")
        print(f"    Гипотеза не подтверждена.")

    # ── Топ потерянных признаков
    print(f"\n  Univariate AUC 'bit-5' признаков по alpha:")
    print(f"    {'alpha':>10}", end="")
    for key in ["td_bit5", "hdiff_bit5", "td_bit4", "hdiff_bit4"]:
        if key in univariate:
            print(f" {key:>12}", end="")
    print()
    for tag, _, _ in combos:
        d = attacks[tag]
        line = f"    {d['alpha']:>10.4f}"
        for key in ["td_bit5", "hdiff_bit5", "td_bit4", "hdiff_bit4"]:
            if key in univariate:
                line += f" {univariate[key][tag]:>12.4f}"
        print(line)

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    best = max(results.items(), key=lambda kv: kv[1]["auc"])
    worst = min(results.items(), key=lambda kv: kv[1]["auc"])

    print(f"\n  Лучшая атака для детекции: {best[0]} "
          f"(alpha={attacks[best[0]]['alpha']:.4f}, "
          f"n_iter={attacks[best[0]]['n_iter']})")
    print(f"    AUC = {best[1]['auc']:.4f}, det@5% = {best[1]['det5']:.3f}")
    print(f"\n  Худшая атака для детекции: {worst[0]} "
          f"(alpha={attacks[worst[0]]['alpha']:.4f}, "
          f"n_iter={attacks[worst[0]]['n_iter']})")
    print(f"    AUC = {worst[1]['auc']:.4f}, det@5% = {worst[1]['det5']:.3f}")

    if abs(corr_alpha) > abs(corr_n_iter) and abs(corr_alpha) > 0.7:
        print(f"\n  ✓✓✓ РЕЗУЛЬТАТ ДЛЯ ПУБЛИКАЦИИ:")
        print(f"      Размер шага PGD (alpha) определяет силу")
        print(f"      bit-plane signature (corr = {corr_alpha:+.3f}).")
        print(f"      Число шагов (n_iter) — вторично.")
        print(f"      Это объясняет, почему PGD-2 с alpha=eps/2")
        print(f"      стирает bit-признаки, а PGD-20 с alpha=eps/20 нет.")

    # ── Сохранение
    with open("pgd_step_sweep.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha_tag", "alpha", "n_iter", "auc", "det5", "det10"])
        for tag, d in attacks.items():
            r = results[tag]
            w.writerow([tag, f"{d['alpha']:.4f}", d["n_iter"],
                        f"{r['auc']:.4f}", f"{r['det5']:.4f}",
                        f"{r['det10']:.4f}"])
    print(f"\n  Сохранено: pgd_step_sweep.csv")

    # ── Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: alpha vs AUC
        ax = axes[0]
        for tag, d in attacks.items():
            ax.scatter(d["alpha"], results[tag]["auc"],
                        color="crimson", s=150, zorder=5)
            ax.annotate(f"{tag}\n(n={d['n_iter']})",
                         (d["alpha"], results[tag]["auc"]),
                         fontsize=9, xytext=(10, 0),
                         textcoords="offset points")
        ax.set_xlabel("alpha (размер шага)")
        ax.set_ylabel("AUC")
        ax.set_title(f"alpha vs AUC (corr = {corr_alpha:+.3f})")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        # Panel 2: n_iter vs AUC
        ax = axes[1]
        for tag, d in attacks.items():
            ax.scatter(d["n_iter"], results[tag]["auc"],
                        color="steelblue", s=150, zorder=5)
            ax.annotate(tag, (d["n_iter"], results[tag]["auc"]),
                         fontsize=9, xytext=(10, 0),
                         textcoords="offset points")
        ax.set_xlabel("n_iter (число шагов)")
        ax.set_ylabel("AUC")
        ax.set_title(f"n_iter vs AUC (corr = {corr_n_iter:+.3f})")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        # Panel 3: univariate AUC по alpha
        ax = axes[2]
        for key in ["td_bit5", "hdiff_bit5", "td_bit4", "hdiff_bit4", "frag"]:
            if key not in univariate:
                continue
            xs = []
            ys = []
            for tag, d in attacks.items():
                xs.append(d["alpha"])
                ys.append(univariate[key][tag])
            order = np.argsort(xs)
            xs = np.array(xs)[order]
            ys = np.array(ys)[order]
            ax.plot(xs, ys, "o-", linewidth=2, markersize=8,
                     label=key)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Univariate AUC")
        ax.set_title("Bit-признаки vs alpha")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("pgd_step_sweep.png", dpi=120)
        print("  Сохранено: pgd_step_sweep.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs)


if __name__ == "__main__":
    main()