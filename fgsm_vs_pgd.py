"""
fgsm_vs_pgd.py — что PGD стирает, что FGSM оставляет?

Наблюдение: наш метод даёт
    FGSM: AUC = 0.994
    PGD:  AUC = 0.940

Разница 0.054 — это научный вопрос. Что именно PGD удаляет
из bit-признаков, что FGSM оставляет?

План:
  1. Обучить классификатор на FGSM-примерах, тестить на PGD.
     Если AUC упадёт до 0.5 — FGSM-признаки специфичны.
     Если останется >0.85 — признак общий.

  2. Декомпозиция по признакам:
     - univariate AUC для каждого из 44 признаков отдельно
       (FGSM vs PGD)
     - найти признаки, которые "работают на FGSM, но не на PGD"

  3. Итеративная атака: FGSM-1, FGSM-2, PGD-5, PGD-10, PGD-20
     - как падает AUC с ростом числа шагов?

  4. Проверка гипотез:
     - H1: PGD стирает td_bit5 (transition density 5-го бита)
     - H2: PGD стирает hdiff_bit5
     - H3: PGD оставляет только low-frequency признаки

Запуск:
    python fgsm_vs_pgd.py --n_benign 500
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

def fgsm_attack(model, x, y, eps=0.05):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).detach().clamp(0, 1)


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


# ── Утилиты ─────────────────────────────────────────────────

def univariate_auc(feats_b, feats_a, key):
    """AUC для одного признака (правильный, через sklearn)."""
    from sklearn.metrics import roc_auc_score
    fb = np.array([f[key] for f in feats_b])
    fa = np.array([f[key] for f in feats_a])
    y = np.concatenate([np.zeros(len(fb)), np.ones(len(fa))])
    scores = np.concatenate([fb, fa])
    auc = roc_auc_score(y, scores)
    # Направление: adversarial чаще меньше
    if fa.mean() < fb.mean():
        return auc
    return 1 - auc


# ── Основной эксперимент ────────────────────────────────────

def run(n_benign=500, n_workers=4):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    print("=" * 80)
    print(f"FGSM vs PGD decomposition — CIFAR-10")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    model, x_all, y_all = load_all()
    total = len(x_all)

    n_train = min(int(0.6 * n_benign), int(0.6 * total))
    n_calib = min(int(0.2 * n_benign), int(0.2 * total))
    n_test = min(n_benign - n_train - n_calib, total - n_train - n_calib)

    print(f"\n[0] Split: train={n_train}, calib={n_calib}, test={n_test}")

    x_b_train = x_all[:n_train].to(DEVICE)
    y_b_train = y_all[:n_train].to(DEVICE)
    x_b_calib = x_all[n_train:n_train + n_calib].to(DEVICE)
    y_b_calib = y_all[n_train:n_train + n_calib].to(DEVICE)
    x_b_test = x_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)
    y_b_test = y_all[n_train + n_calib:n_train + n_calib + n_test].to(DEVICE)

    # ── Генерация 5 типов атак
    print(f"\n[1] Генерация 5 типов атак:")
    attacks = {}

    # FGSM-1
    t0 = time.time()
    attacks["FGSM"] = {
        "train": fgsm_attack(model, x_b_train.clone(), y_b_train),
        "calib": fgsm_attack(model, x_b_calib.clone(), y_b_calib),
        "test":  fgsm_attack(model, x_b_test.clone(), y_b_test),
    }
    print(f"    FGSM:      {time.time()-t0:.1f}s")

    # PGD с разным числом шагов
    for n_iter, name in [(2, "PGD-2"), (5, "PGD-5"),
                          (10, "PGD-10"), (20, "PGD-20")]:
        t0 = time.time()
        attacks[name] = {
            "train": pgd_attack(model, x_b_train.clone(), y_b_train,
                                 n_iter=n_iter),
            "calib": pgd_attack(model, x_b_calib.clone(), y_b_calib,
                                 n_iter=n_iter),
            "test":  pgd_attack(model, x_b_test.clone(), y_b_test,
                                 n_iter=n_iter),
        }
        print(f"    {name}:    {time.time()-t0:.1f}s")

    # ASR для каждой атаки
    print(f"\n[2] ASR на test:")
    with torch.no_grad():
        pred_b = model(x_b_test).argmax(1)
        for name, data in attacks.items():
            pred_a = model(data["test"]).argmax(1)
            asr = ((pred_b != pred_a)
                    & (pred_b == y_b_test)).float().mean().item()
            print(f"    {name:<10} ASR = {asr:.3f}")

    # ── Извлечение признаков
    print(f"\n[3] Извлечение bit-признаков ...")
    t0 = time.time()
    fb_train = extract_features(x_b_train, n_workers)
    fb_calib = extract_features(x_b_calib, n_workers)
    fb_test = extract_features(x_b_test, n_workers)

    fa_train = {}
    fa_calib = {}
    fa_test = {}
    for name, data in attacks.items():
        fa_train[name] = extract_features(data["train"], n_workers)
        fa_calib[name] = extract_features(data["calib"], n_workers)
        fa_test[name] = extract_features(data["test"], n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    print(f"    Признаков: {len(keys)}")

    def to_X(feats):
        return np.array([[f[k] for k in keys] for f in feats])

    # ── Эксперимент 1: декомпозиция по univariate AUC
    print(f"\n[4] Univariate AUC по каждому признаку:")
    print(f"    {'feature':<16}", end="")
    for name in attacks:
        print(f" {name:>9}", end="")
    print(f"  {'max diff':>10}")
    print("    " + "-" * (16 + 10 * len(attacks) + 12))

    univariate = {}
    for key in keys:
        row = []
        for name in attacks:
            auc = univariate_auc(fb_test, fa_test[name], key)
            row.append(auc)
        univariate[key] = dict(zip(attacks.keys(), row))
        max_diff = max(row) - min(row)
        line = f"    {key:<16}"
        for name in attacks:
            marker = " "
            if univariate[key][name] < 0.55 and \
               univariate[key]["FGSM"] > 0.75:
                marker = "!"
            line += f" {univariate[key][name]:>8.4f}{marker}"
        line += f"  {max_diff:>10.4f}"
        print(line)

    # ── Эксперимент 2: FGSM vs PGD — какие признаки теряют силу
    print(f"\n[5] Признаки, которые FGSM ловит, а PGD нет:")
    print(f"    {'feature':<16} {'FGSM':>8} {'PGD-20':>8} {'Δ':>8}")
    print("    " + "-" * 42)

    lost_features = []
    for key in keys:
        auc_fgsm = univariate[key]["FGSM"]
        auc_pgd = univariate[key]["PGD-20"]
        delta = auc_fgsm - auc_pgd
        if auc_fgsm > 0.70 and auc_pgd < 0.60:
            lost_features.append({
                "key": key, "fgsm": auc_fgsm,
                "pgd": auc_pgd, "delta": delta,
            })

    lost_features.sort(key=lambda x: -x["delta"])
    for r in lost_features[:15]:
        print(f"    {r['key']:<16} {r['fgsm']:>8.4f} {r['pgd']:>8.4f} "
              f"{r['delta']:>+8.4f}")

    if not lost_features:
        print(f"    Нет признаков с таким паттерном")
        print(f"    → FGSM и PGD работают схоже на всех признаках")

    # ── Эксперимент 3: Обобщение между атаками
    print(f"\n[6] Обобщение: обучение на FGSM → тест на PGD:")
    print(f"    {'train':<10} {'test':<10} {'AUC':>8} {'det@5%':>8}")
    print("    " + "-" * 40)

    # Обучение на FGSM
    X_train = np.vstack([to_X(fb_train), to_X(fa_train["FGSM"])])
    y_train = np.concatenate([np.zeros(len(fb_train)),
                               np.ones(len(fa_train["FGSM"]))])

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Threshold из calib
    prob_b_calib = clf.predict_proba(to_X(fb_calib))[:, 1]
    sorted_calib = np.sort(prob_b_calib)[::-1]
    idx5 = min(int(0.05 * len(sorted_calib)), len(sorted_calib) - 1)
    thr5 = sorted_calib[idx5]

    cross_results = {}
    for name in attacks:
        prob_b_test = clf.predict_proba(to_X(fb_test))[:, 1]
        prob_a_test = clf.predict_proba(to_X(fa_test[name]))[:, 1]

        y_test = np.concatenate([np.zeros(len(prob_b_test)),
                                  np.ones(len(prob_a_test))])
        scores = np.concatenate([prob_b_test, prob_a_test])
        auc = roc_auc_score(y_test, scores)

        fpr5 = float((prob_b_test > thr5).mean())
        det5 = float((prob_a_test > thr5).mean())

        cross_results[name] = {"auc": auc, "det5": det5}
        marker = ""
        if name == "FGSM":
            marker = " ← обучение"
        elif auc < 0.7:
            marker = " ← потеря"
        print(f"    FGSM       {name:<10} {auc:>8.4f} {det5:>8.3f}{marker}")

    # ── Эксперимент 4: что если обучать на PGD и тестить на FGSM?
    print(f"\n[7] Обратное: обучение на PGD → тест на FGSM:")
    X_train = np.vstack([to_X(fb_train), to_X(fa_train["PGD-20"])])
    y_train = np.concatenate([np.zeros(len(fb_train)),
                               np.ones(len(fa_train["PGD-20"]))])

    clf2 = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42,
    )
    clf2.fit(X_train, y_train)

    prob_b_calib2 = clf2.predict_proba(to_X(fb_calib))[:, 1]
    sorted_calib2 = np.sort(prob_b_calib2)[::-1]
    idx5_2 = min(int(0.05 * len(sorted_calib2)), len(sorted_calib2) - 1)
    thr5_2 = sorted_calib2[idx5_2]

    print(f"    {'train':<10} {'test':<10} {'AUC':>8} {'det@5%':>8}")
    print("    " + "-" * 40)

    reverse_results = {}
    for name in attacks:
        prob_b_test = clf2.predict_proba(to_X(fb_test))[:, 1]
        prob_a_test = clf2.predict_proba(to_X(fa_test[name]))[:, 1]

        y_test = np.concatenate([np.zeros(len(prob_b_test)),
                                  np.ones(len(prob_a_test))])
        scores = np.concatenate([prob_b_test, prob_a_test])
        auc = roc_auc_score(y_test, scores)

        fpr5 = float((prob_b_test > thr5_2).mean())
        det5 = float((prob_a_test > thr5_2).mean())

        reverse_results[name] = {"auc": auc, "det5": det5}
        marker = ""
        if name == "PGD-20":
            marker = " ← обучение"
        elif auc < 0.7:
            marker = " ← потеря"
        print(f"    PGD-20     {name:<10} {auc:>8.4f} {det5:>8.3f}{marker}")

    # ── Итог
    print(f"\n{'=' * 80}")
    print("АНАЛИЗ")
    print(f"{'=' * 80}")

    print(f"\n  Обобщение FGSM → PGD:")
    print(f"    AUC(FGSM→FGSM) = {cross_results['FGSM']['auc']:.4f}")
    print(f"    AUC(FGSM→PGD)  = {cross_results['PGD-20']['auc']:.4f}")
    print(f"    Разница: {cross_results['FGSM']['auc'] - cross_results['PGD-20']['auc']:+.4f}")

    print(f"\n  Обобщение PGD → FGSM:")
    print(f"    AUC(PGD→PGD)   = {reverse_results['PGD-20']['auc']:.4f}")
    print(f"    AUC(PGD→FGSM)  = {reverse_results['FGSM']['auc']:.4f}")
    print(f"    Разница: {reverse_results['PGD-20']['auc'] - reverse_results['FGSM']['auc']:+.4f}")

    # Гипотеза о битах
    print(f"\n  Гипотеза о bit-плоскостях:")
    bit_losses = []
    for key in keys:
        if key.startswith("td_bit") or key.startswith("hdiff_bit"):
            loss = univariate[key]["FGSM"] - univariate[key]["PGD-20"]
            bit_losses.append((key, loss))

    bit_losses.sort(key=lambda x: -x[1])
    print(f"    Top-5 потерявших признаков:")
    for key, loss in bit_losses[:5]:
        print(f"      {key:<16} FGSM→PGD потеря = {loss:+.4f}")

    # Сохранение
    with open("fgsm_vs_pgd.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["feature"] + list(attacks.keys())
        w.writerow(header)
        for key in keys:
            row = [key] + [f"{univariate[key][n]:.4f}"
                            for n in attacks]
            w.writerow(row)
    print(f"\n  Сохранено: fgsm_vs_pgd.csv")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: univariate AUC для top признаков
        ax = axes[0]
        top_keys = sorted(keys,
                            key=lambda k: -univariate[k]["FGSM"])[:12]
        x = np.arange(len(top_keys))
        w = 0.15
        for i, name in enumerate(attacks):
            vals = [univariate[k][name] for k in top_keys]
            ax.barh(x + i * w, vals, w, label=name)
        ax.set_yticks(x)
        ax.set_yticklabels(top_keys, fontsize=8)
        ax.set_xlabel("AUC")
        ax.set_title("Univariate AUC: FGSM vs PGD")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="x")

        # Panel 2: scatter FGSM vs PGD
        ax = axes[1]
        for key in keys:
            xf = univariate[key]["FGSM"]
            xp = univariate[key]["PGD-20"]
            color = "crimson" if abs(xf - xp) > 0.15 else "steelblue"
            ax.scatter(xf, xp, color=color, alpha=0.7, s=30)
            if abs(xf - xp) > 0.20:
                ax.annotate(key, (xf, xp), fontsize=7)
        ax.plot([0.4, 1.0], [0.4, 1.0], "k--", alpha=0.3)
        ax.set_xlabel("AUC на FGSM")
        ax.set_ylabel("AUC на PGD-20")
        ax.set_title("Что FGSM ловит, PGD стирает")
        ax.grid(alpha=0.3)

        # Panel 3: cross-attack generalization
        ax = axes[2]
        names = list(attacks.keys())
        fgsm_trained = [cross_results[n]["auc"] for n in names]
        pgd_trained = [reverse_results[n]["auc"] for n in names]
        x = np.arange(len(names))
        ax.plot(x, fgsm_trained, "o-", color="steelblue",
                linewidth=2, markersize=10, label="trained on FGSM")
        ax.plot(x, pgd_trained, "s-", color="crimson",
                linewidth=2, markersize=10, label="trained on PGD-20")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, fontsize=9)
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Cross-attack generalisation")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("fgsm_vs_pgd.png", dpi=120)
        print("  Сохранено: fgsm_vs_pgd.png")
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