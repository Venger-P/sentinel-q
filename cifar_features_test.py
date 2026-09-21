"""
cifar_features_test.py — быстрое сравнение наборов признаков.

Проблема: SBG_GREEDY_FEATURES (10 признаков без frag) даёт AUC=0.90.
А в cifar_bitwise.py набор frag+td_bit давал AUC=0.9693.

Проверяем 6 наборов × 3 модели × 2 режима train = 36 комбинаций.

Ключ: базовый frag (на полном изображении) + td_bit* на битовых слоях.

Запуск:
    python cifar_features_test.py --n_benign 500
"""

import argparse
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


# ── ПОЛНЫЙ набор признаков ─────────────────────────────────

def all_features(img_np):
    """Все возможные bit-признаки + frag + H + nu."""
    out = {}
    arr = (img_np * 255).astype(np.uint8)

    # Базовые признаки на полном байте
    full = arr.tobytes()
    out["frag"] = fragility(full, "zlib")
    out["frag_b"] = fragility(full, "bz2")
    out["H"] = shannon_entropy(full)
    out["nu"] = len(np.unique(np.frombuffer(full, dtype=np.uint8)))

    # Бит-плоскости
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
        out[f"hdiff_bit{bit}"] = float(
            np.abs(bp[:, 1:] - bp[:, :-1]).mean()
        )
        out[f"vdiff_bit{bit}"] = float(
            np.abs(bp[1:, :] - bp[:-1, :]).mean()
        )

    return out


# Определяем 6 наборов
def get_feature_sets(available_keys):
    sbf_greedy = [
        "td_bit5", "hdiff_bit5", "td_bit4", "td_bit6", "hdiff_bit4",
        "frag_bit4", "td_bit0", "frag_bit3", "td_bit2", "H_bit5",
    ]
    frag_td = ["frag", "frag_b"] + [f"td_bit{i}" for i in range(8)]
    frag_hdiff = ["frag", "frag_b"] + [f"hdiff_bit{i}" for i in range(8)]
    frag_bit_all = ["frag", "frag_b", "H", "nu"] + \
                   [f"frag_bit{i}" for i in range(8)]
    frag_td_hdiff = ["frag", "frag_b"] + \
                    [f"td_bit{i}" for i in range(8)] + \
                    [f"hdiff_bit{i}" for i in range(8)]
    all_keys = list(available_keys)

    sets = {
        "SBF greedy (10)":       sbf_greedy,
        "frag + td_bit (10)":    frag_td,
        "frag + hdiff_bit (10)": frag_hdiff,
        "frag + frag_bit (12)":  frag_bit_all,
        "frag + td + hdiff (18)": frag_td_hdiff,
        "all (52)":              all_keys,
    }
    # Фильтруем те, что содержат только доступные ключи
    return {name: [k for k in keys if k in available_keys]
            for name, keys in sets.items()}


def extract_all(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(img):
        return all_features(img)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(one, [arr[i] for i in range(len(arr))]))
    return results


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


def run(n_benign=500, n_workers=4, target_fpr=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline

    print("=" * 80)
    print(f"Сравнение наборов признаков — n_benign={n_benign}")
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

    # Adv: 30 для train, все для calib/test
    n_adv_train = max(1, n_train // 10)
    x_a_train = x_b_train[:n_adv_train].clone()
    y_a_train = y_b_train[:n_adv_train]

    print(f"    Train: {n_train} b + {n_adv_train} a")
    print(f"    Calib: {n_calib} b + {n_calib} a")
    print(f"    Test:  {n_test} b + {n_test} a")

    # PGD
    print(f"\n[1] PGD-20 ...")
    t0 = time.time()
    x_a_train = pgd_attack(model, x_a_train, y_a_train)
    x_a_calib = pgd_attack(model, x_b_calib, y_b_calib)
    x_a_test = pgd_attack(model, x_b_test, y_b_test)
    print(f"    {time.time()-t0:.1f}s")

    # Извлечение полного набора для всех
    print(f"\n[2] Извлечение всех признаков ...")
    t0 = time.time()
    fb_train = extract_all(x_b_train, n_workers)
    fa_train = extract_all(x_a_train, n_workers)
    fb_calib = extract_all(x_b_calib, n_workers)
    fa_calib = extract_all(x_a_calib, n_workers)
    fb_test = extract_all(x_b_test, n_workers)
    fa_test = extract_all(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    available_keys = sorted(fb_train[0].keys())
    print(f"    Доступно признаков: {len(available_keys)}")

    feature_sets = get_feature_sets(available_keys)

    # ── Модели
    classifiers = {
        "LR": make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=5000),
        ),
        "RF": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            random_state=42, n_jobs=-1,
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        ),
    }

    print(f"\n[3] Тестируем {len(feature_sets)} наборов × "
          f"{len(classifiers)} моделей")
    print(f"\n    {'features':<26} {'model':<5} {'AUC':>7} "
          f"{'FPR@5%':>8} {'det@5%':>8}")
    print("    " + "-" * 58)

    results = []
    for fname, keys in feature_sets.items():
        if not keys:
            continue

        def to_X(feature_list):
            return np.array([[f[k] for k in keys] for f in feature_list])

        X_train = np.vstack([to_X(fb_train), to_X(fa_train)])
        y_train = np.concatenate([np.zeros(n_train),
                                   np.ones(n_adv_train)])
        X_calib_b = to_X(fb_calib)
        X_test_b = to_X(fb_test)
        X_test_a = to_X(fa_test)

        y_test = np.concatenate([np.zeros(n_test), np.ones(n_test)])

        for mname, clf in classifiers.items():
            try:
                clf.fit(X_train, y_train)
                prob_b_calib = clf.predict_proba(X_calib_b)[:, 1]
                prob_b_test = clf.predict_proba(X_test_b)[:, 1]
                prob_a_test = clf.predict_proba(X_test_a)[:, 1]

                y_scores = np.concatenate([prob_b_test, prob_a_test])
                auc = roc_auc_score(y_test, y_scores)

                # Калибровка
                sorted_calib = np.sort(prob_b_calib)[::-1]
                idx = min(int(target_fpr * len(sorted_calib)),
                           len(sorted_calib) - 1)
                thr = sorted_calib[idx]
                fpr_5 = float((prob_b_test > thr).mean())
                det_5 = float((prob_a_test > thr).mean())

                results.append({
                    "features": fname, "model": mname,
                    "n_keys": len(keys),
                    "auc": auc, "fpr_5": fpr_5, "det_5": det_5,
                })

                marker = ""
                if fpr_5 <= 0.07 and det_5 >= 0.70:
                    marker = " ✓✓✓"
                elif fpr_5 <= 0.07 and det_5 >= 0.50:
                    marker = " ✓"

                print(f"    {fname:<26} {mname:<5} {auc:>7.4f} "
                      f"{fpr_5:>8.3f} {det_5:>8.3f}{marker}")
            except Exception as e:
                print(f"    {fname:<26} {mname:<5} ошибка: {e}")

    # ── Итог
    print(f"\n{'=' * 80}")
    print("TOP-5 по det@FPR=5%:")
    print(f"{'=' * 80}")
    sorted_r = sorted(results, key=lambda r: -r["det_5"])
    for i, r in enumerate(sorted_r[:5]):
        print(f"  {i+1}. {r['features']:<26} {r['model']:<5} "
              f"n={r['n_keys']:>2}  AUC={r['auc']:.4f}  "
              f"FPR={r['fpr_5']:.3f}  det={r['det_5']:.3f}")

    print(f"\nTOP-5 по AUC:")
    sorted_auc = sorted(results, key=lambda r: -r["auc"])
    for i, r in enumerate(sorted_auc[:5]):
        print(f"  {i+1}. {r['features']:<26} {r['model']:<5} "
              f"AUC={r['auc']:.4f}")

    # Лучший по комбинированному критерию
    best = max(results, key=lambda r: r["det_5"] - r["fpr_5"])
    print(f"\nЛучший по det-FPR: {best['features']} + {best['model']}")
    print(f"  AUC={best['auc']:.4f}, FPR={best['fpr_5']:.3f}, "
          f"det={best['det_5']:.3f}")

    if best["det_5"] >= 0.75 and best["fpr_5"] <= 0.07:
        print(f"\n✓✓✓ ЦЕЛЬ ДОСТИГНУТА")
    elif best["det_5"] >= 0.60:
        print(f"\n✓ Хорошо, но не production")
    else:
        print(f"\n~ Требуется дальнейшая настройка")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()

    run(n_benign=args.n_benign, n_workers=args.jobs,
        target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()