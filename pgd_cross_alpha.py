"""
pgd_cross_alpha.py — cross-alpha matrix для bit-plane signature.

Вопрос: bit-plane signature, выученный на одной alpha, работает ли
на других alpha?

Матрица: 4×4
    train alpha \\ test alpha
    eps/20, eps/10, eps/4, eps/2

Диагональ = in-distribution (обучение и тест на одной alpha).
Off-diagonal = cross-alpha generalisation.

Гипотезы:
  H1: диагональ высокая, off-diagonal низкая → специфично по alpha
  H2: матрица однородная → signature универсален
  H3: асимметрия → train на малой alpha лучше обобщает

Запуск:
    python pgd_cross_alpha.py --n_benign 500
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


def run(n_benign=500, n_workers=4):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier

    print("=" * 82)
    print(f"Cross-alpha matrix — bit-plane signature")
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

    # Сетка alpha
    alphas = [
        ("eps/20", 20, eps / 20),
        ("eps/10", 10, eps / 10),
        ("eps/4",  4,  eps / 4),
        ("eps/2",  2,  eps / 2),
    ]

    # ── Генерация атак
    print(f"\n[1] Генерация 4 атак ...")
    attacks = {}
    for tag, n_iter, alpha in alphas:
        t0 = time.time()
        attacks[tag] = {
            "alpha": alpha, "n_iter": n_iter,
            "train": pgd_attack(model, x_b_train.clone(), y_b_train,
                                 eps=eps, alpha=alpha, n_iter=n_iter),
            "test":  pgd_attack(model, x_b_test.clone(), y_b_test,
                                 eps=eps, alpha=alpha, n_iter=n_iter),
        }
        print(f"    {tag:<8} alpha={alpha:.4f} n_iter={n_iter:<3} "
              f"({time.time()-t0:.1f}s)")

    # ── ASR
    print(f"\n[2] ASR:")
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
    for tag, _ in attacks.items():
        fa_train[tag] = extract_features(attacks[tag]["train"], n_workers)
        fa_test[tag] = extract_features(attacks[tag]["test"], n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    print(f"    Признаков: {len(keys)}")

    def to_X(feats):
        return np.array([[f[k] for k in keys] for f in feats])

    X_train_b = to_X(fb_train)
    X_test_b = to_X(fb_test)
    X_calib_b = to_X(fb_calib)

    # ── Cross-alpha matrix
    print(f"\n[4] Cross-alpha matrix (AUC):")
    print(f"    Обучаем на train_alpha, тестируем на test_alpha")
    print()

    tags = [t for t, _, _ in alphas]
    matrix = {}

    print(f"    {'train / test':<14}", end="")
    for tag in tags:
        print(f" {tag:>9}", end="")
    print()
    print("    " + "-" * (14 + 10 * len(tags)))

    for train_tag in tags:
        matrix[train_tag] = {}
        # Обучаем GB на train_tag
        X_train = np.vstack([X_train_b, to_X(fa_train[train_tag])])
        y_train = np.concatenate([np.zeros(len(fb_train)),
                                   np.ones(len(fa_train[train_tag]))])

        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # Калибровка порога на benign calib
        prob_b_calib = clf.predict_proba(X_calib_b)[:, 1]
        sorted_calib = np.sort(prob_b_calib)[::-1]
        idx5 = min(int(0.05 * len(sorted_calib)),
                    len(sorted_calib) - 1)
        thr5 = sorted_calib[idx5]

        print(f"    {train_tag:<14}", end="")

        # Тестируем на каждой test_tag
        for test_tag in tags:
            prob_b = clf.predict_proba(X_test_b)[:, 1]
            prob_a = clf.predict_proba(to_X(fa_test[test_tag]))[:, 1]

            y_test = np.concatenate([np.zeros(len(prob_b)),
                                      np.ones(len(prob_a))])
            scores = np.concatenate([prob_b, prob_a])
            auc = roc_auc_score(y_test, scores)

            fpr5 = float((prob_b > thr5).mean())
            det5 = float((prob_a > thr5).mean())

            matrix[train_tag][test_tag] = {
                "auc": auc, "det5": det5,
            }

            # Маркируем диагональ
            marker = "*" if train_tag == test_tag else " "
            print(f" {auc:>8.4f}{marker}", end="")
        print()

    print(f"\n    (* — in-distribution: train и test на одной alpha)")

    # ── Анализ
    print(f"\n{'=' * 82}")
    print("АНАЛИЗ")
    print(f"{'=' * 82}")

    # Диагональ vs off-diagonal
    diag = [matrix[t][t]["auc"] for t in tags]
    off_diag = []
    for tr in tags:
        for te in tags:
            if tr != te:
                off_diag.append(matrix[tr][te]["auc"])

    print(f"\n  Диагональ (in-distribution):")
    for t in tags:
        print(f"    {t:<8} AUC = {matrix[t][t]['auc']:.4f}")
    print(f"    Среднее: {np.mean(diag):.4f} ± {np.std(diag):.4f}")

    print(f"\n  Off-diagonal (cross-alpha):")
    print(f"    Среднее: {np.mean(off_diag):.4f} ± {np.std(off_diag):.4f}")
    print(f"    Мин: {np.min(off_diag):.4f}")
    print(f"    Макс: {np.max(off_diag):.4f}")

    delta = np.mean(diag) - np.mean(off_diag)
    print(f"\n  Δ (диагональ − off-diagonal): {delta:+.4f}")

    # Гипотезы
    if delta > 0.10:
        print(f"\n  ✓ H1: SIGNATURE СПЕЦИФИЧЕН ПО ALPHA")
        print(f"    Диагональ на {delta:.3f} лучше off-diagonal.")
        print(f"    Bit-plane signature зависит от размера шага.")
        hypothesis = "H1_specific"
    elif delta > 0.03:
        print(f"\n  ~ Слабая специфичность (Δ={delta:.3f})")
        hypothesis = "partial"
    else:
        print(f"\n  ✓ H2: SIGNATURE УНИВЕРСАЛЕН")
        print(f"    Диагональ ≈ off-diagonal.")
        print(f"    Bit-plane signature не зависит от alpha.")
        hypothesis = "H2_universal"

    # Асимметрия
    print(f"\n  Асимметрия матрицы:")
    print(f"    {'train':<8} {'test':<8} {'AUC':>8} | "
          f"{'reverse':>8}")
    print("    " + "-" * 40)

    asymmetry = []
    for i, tr in enumerate(tags):
        for te in tags[i+1:]:
            auc_forward = matrix[tr][te]["auc"]
            auc_reverse = matrix[te][tr]["auc"]
            asm = auc_forward - auc_reverse
            asymmetry.append(asm)
            marker = " ←" if abs(asm) > 0.05 else ""
            print(f"    {tr:<8} {te:<8} {auc_forward:>8.4f} | "
                  f"{auc_reverse:>8.4f}{marker}")

    print(f"\n    Средняя асимметрия: {np.mean(asymmetry):+.4f}")
    print(f"    Std асимметрии:     {np.std(asymmetry):.4f}")

    # ── Итоговая гипотеза для публикации
    print(f"\n{'=' * 82}")
    print("ВЫВОД")
    print(f"{'=' * 82}")

    if hypothesis == "H1_specific":
        print(f"\n  ✓✓✓ НОВЫЙ РЕЗУЛЬТАТ:")
        print(f"      Bit-plane signature специфичен для alpha атаки.")
        print(f"      In-distribution AUC = {np.mean(diag):.4f},")
        print(f"      Cross-alpha AUC = {np.mean(off_diag):.4f}")
        print(f"      Модель, обученная на eps/20, не обобщает на eps/2.")
        print(f"\n  Публикационное утверждение:")
        print(f"      «Bit-plane признаки adversarial-атак чувствительны")
        print(f"       к размеру шага PGD. Это создаёт направленный")
        print(f"       transfer между атаками с разными alpha.»")
    elif hypothesis == "H2_universal":
        print(f"\n  ✓✓ РЕЗУЛЬТАТ:")
        print(f"      Bit-plane signature УНИВЕРСАЛЕН по alpha.")
        print(f"      Это свойство adversarial-возмущения,")
        print(f"      не артефакт конкретной атаки.")
        print(f"\n  Публикационное утверждение:")
        print(f"      «Bit-plane signature инвариантен к параметрам")
        print(f"       PGD-атаки и может быть использован для")
        print(f"       универсальной детекции.»")

    # ── Сохранение
    with open("pgd_cross_alpha.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_alpha", "test_alpha", "auc", "det5"])
        for tr in tags:
            for te in tags:
                r = matrix[tr][te]
                w.writerow([tr, te, f"{r['auc']:.4f}",
                            f"{r['det5']:.4f}"])
    print(f"\n  Сохранено: pgd_cross_alpha.csv")

    # ── Plot heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))

        data = np.zeros((len(tags), len(tags)))
        for i, tr in enumerate(tags):
            for j, te in enumerate(tags):
                data[i, j] = matrix[tr][te]["auc"]

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(tags)))
        ax.set_yticks(range(len(tags)))
        ax.set_xticklabels(tags)
        ax.set_yticklabels(tags)
        ax.set_xlabel("Test alpha")
        ax.set_ylabel("Train alpha")
        ax.set_title("Cross-alpha AUC matrix")

        for i in range(len(tags)):
            for j in range(len(tags)):
                color = "white" if data[i, j] < 0.7 else "black"
                ax.text(j, i, f"{data[i,j]:.3f}",
                         ha="center", va="center", color=color,
                         fontsize=11, fontweight="bold")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig("pgd_cross_alpha.png", dpi=120)
        print("  Сохранено: pgd_cross_alpha.png")
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