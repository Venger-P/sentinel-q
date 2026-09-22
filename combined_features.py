"""
combined_features.py — амплитуда + bit-структура.

Открытие: FAB не детектируется bit-признаками, потому что создаёт
возмущение в 6 раз меньше (linf = 0.0085 против 0.05).

Гипотеза: если добавить признаки амплитуды возмущения к bit-признакам,
FAB тоже детектируется.

Признаки:
  Бит-структура (44):
    - frag, H, nu
    - frag_bit, H_bit, td_bit, hdiff_bit, vdiff_bit

  Амплитуда (7):
    - linf: max|δ|
    - l2: ||δ||₂
    - l0: число пикселей с |δ| > 1/255
    - mean_abs: среднее |δ|
    - top1pct: top-1% амплитуд
    - peak_ratio: linf / mean|δ|
    - entropy_pert: энтропия |δ|

Но: амплитуда требует знания оригинала x. Для детекции это
означает необходимость сравнения с оригиналом — что означает
не чистый "чёрный ящик", а side-channel с эталоном.

Запуск:
    python combined_features.py --n_benign 500
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


def perturbation_features(x_orig_np, x_adv_np):
    """
    Признаки возмущения. Требуют ОРИГИНАЛА x.

    x_orig_np, x_adv_np: (C, H, W) float [0, 1].
    """
    delta = x_adv_np - x_orig_np
    abs_delta = np.abs(delta.flatten())

    linf = float(abs_delta.max())
    l2 = float(np.sqrt((delta.flatten() ** 2).sum()))
    l0 = int((abs_delta > 1.0 / 255).sum())
    mean_abs = float(abs_delta.mean())

    n_top = max(1, int(0.01 * len(abs_delta)))
    top1pct = float(np.sort(abs_delta)[-n_top:].mean())

    peak_ratio = linf / max(mean_abs, 1e-9)

    hist, _ = np.histogram(abs_delta, bins=256, range=(0, 1))
    hist = hist / max(hist.sum(), 1)
    probs = hist[hist > 0]
    entropy_pert = float(-np.sum(probs * np.log2(probs)))

    return {
        "pert_linf": linf,
        "pert_l2": l2,
        "pert_l0": float(l0),
        "pert_mean_abs": mean_abs,
        "pert_top1pct": top1pct,
        "pert_peak_ratio": peak_ratio,
        "pert_entropy": entropy_pert,
    }


def extract_all_features(x_tensor, x_orig_tensor, n_workers=4):
    """
    x_tensor: данные (для bit-признаков)
    x_orig_tensor: оригиналы (для амплитудных признаков)
    """
    arr = x_tensor.cpu().numpy()
    arr_orig = x_orig_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def one(args):
        i, img = args
        out = bit_features_one(img)
        pert = perturbation_features(arr_orig[i], img)
        out.update(pert)
        return out

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(one, [(i, arr[i]) for i in range(len(arr))]))


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


def threshold_for_fpr(prob_benign, target_fpr):
    sorted_probs = np.sort(prob_benign)[::-1]
    idx = min(int(target_fpr * len(sorted_probs)),
              len(sorted_probs) - 1)
    return float(sorted_probs[idx])


def run(n_benign=500, n_workers=4, eps=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 82)
    print(f"Combined features: bit-structure + amplitude")
    print(f"Device: {DEVICE}, eps = {eps}")
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

    # ── Генерация 4 атак
    print(f"\n[1] Генерация AutoAttack (4 атаки) ...")
    attack_ids = {
        "APGD-CE":  "apgd-ce",
        "APGD-DLR": "apgd-dlr",
        "FAB":      "fab",
        "Square":   "square",
    }

    attacks = {}
    for name, atk_id in attack_ids.items():
        adversary = AutoAttack(
            model, norm='Linf', eps=eps, version='custom',
            device=DEVICE, verbose=False,
            attacks_to_run=[atk_id],
        )
        t0 = time.time()
        x_train_adv = adversary.run_standard_evaluation(
            x_b_train, y_b_train, bs=100
        )
        x_test_adv = adversary.run_standard_evaluation(
            x_b_test, y_b_test, bs=100
        )

        with torch.no_grad():
            pred_b = model(x_b_test).argmax(1)
            pred_a = model(x_test_adv).argmax(1)
            asr = ((pred_b != pred_a)
                    & (pred_b == y_b_test)).float().mean().item()

        attacks[name] = {
            "train": x_train_adv,
            "test": x_test_adv,
            "asr": asr,
        }
        print(f"    {name:<10} ASR={asr:.3f}  ({time.time()-t0:.1f}s)")

    # ── Извлечение признаков
    print(f"\n[2] Извлечение признаков (bit + amplitude) ...")
    t0 = time.time()
    fb_train = extract_all_features(x_b_train, x_b_train, n_workers)
    fb_test = extract_all_features(x_b_test, x_b_test, n_workers)
    fb_calib = extract_all_features(x_b_calib, x_b_calib, n_workers)

    fa_train = {}
    fa_test = {}
    for name in attacks:
        fa_train[name] = extract_all_features(
            attacks[name]["train"], x_b_train, n_workers
        )
        fa_test[name] = extract_all_features(
            attacks[name]["test"], x_b_test, n_workers
        )
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    bit_keys = [k for k in keys if not k.startswith("pert_")]
    amp_keys = [k for k in keys if k.startswith("pert_")]
    print(f"    Bit-признаков: {len(bit_keys)}")
    print(f"    Amplitude-признаков: {len(amp_keys)}")
    print(f"    Всего: {len(keys)}")

    def to_X(feats, kset=None):
        if kset is None:
            kset = keys
        return np.array([[f[k] for k in kset] for f in feats])

    X_train_b = to_X(fb_train)
    X_test_b = to_X(fb_test)
    X_calib_b = to_X(fb_calib)

    # ── Сравнение: bit-only vs bit+amp vs amp-only
    print(f"\n[3] Сравнение наборов признаков:")
    print(f"    {'attack':<10} {'bit-only':>10} {'bit+amp':>10} "
          f"{'amp-only':>10}")
    print("    " + "-" * 44)

    results = {}
    for name in attacks:
        row = {}

        # Bit-only
        X_tr = np.vstack([to_X(fb_train, bit_keys),
                           to_X(fa_train[name], bit_keys)])
        y_tr = np.concatenate([np.zeros(len(fb_train)),
                                np.ones(len(fa_train[name]))])
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        prob_b = clf.predict_proba(to_X(fb_test, bit_keys))[:, 1]
        prob_a = clf.predict_proba(to_X(fa_test[name], bit_keys))[:, 1]
        y_test = np.concatenate([np.zeros(len(prob_b)),
                                  np.ones(len(prob_a))])
        auc_bit = roc_auc_score(y_test,
                                 np.concatenate([prob_b, prob_a]))

        # Bit + amp
        X_tr_all = np.vstack([to_X(fb_train), to_X(fa_train[name])])
        clf2 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        clf2.fit(X_tr_all, y_tr)
        prob_b2 = clf2.predict_proba(X_test_b)[:, 1]
        prob_a2 = clf2.predict_proba(to_X(fa_test[name]))[:, 1]
        auc_all = roc_auc_score(y_test,
                                 np.concatenate([prob_b2, prob_a2]))

        # Amp-only
        X_tr_amp = np.vstack([to_X(fb_train, amp_keys),
                               to_X(fa_train[name], amp_keys)])
        clf3 = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42,
        )
        clf3.fit(X_tr_amp, y_tr)
        prob_b3 = clf3.predict_proba(to_X(fb_test, amp_keys))[:, 1]
        prob_a3 = clf3.predict_proba(to_X(fa_test[name], amp_keys))[:, 1]
        auc_amp = roc_auc_score(y_test,
                                 np.concatenate([prob_b3, prob_a3]))

        row = {"bit": auc_bit, "all": auc_all, "amp": auc_amp}
        results[name] = row

        marker = ""
        if auc_all > 0.90:
            marker = " ✓✓✓"
        elif auc_all > 0.80:
            marker = " ✓✓"

        print(f"    {name:<10} {auc_bit:>10.4f} {auc_all:>10.4f} "
              f"{auc_amp:>10.4f}{marker}")

    # ── Детальные метрики при FPR=5%
    print(f"\n[4] Detection @ FPR=5% (bit+amp):")
    print(f"    {'attack':<10} {'bit-only':>10} {'bit+amp':>10} "
          f"{'amp-only':>10}")
    print("    " + "-" * 44)

    for name in attacks:
        dets = {}
        for label, kset in [("bit", bit_keys), ("all", keys),
                              ("amp", amp_keys)]:
            X_tr = np.vstack([to_X(fb_train, kset),
                               to_X(fa_train[name], kset)])
            y_tr = np.concatenate([np.zeros(len(fb_train)),
                                    np.ones(len(fa_train[name]))])
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob_b_calib = clf.predict_proba(
                to_X(fb_calib, kset))[:, 1]
            thr = threshold_for_fpr(prob_b_calib, 0.05)
            prob_a = clf.predict_proba(to_X(fa_test[name], kset))[:, 1]
            dets[label] = float((prob_a > thr).mean())

        marker = ""
        if dets["all"] > 0.85:
            marker = " ✓✓✓"
        elif dets["all"] > 0.70:
            marker = " ✓✓"

        print(f"    {name:<10} {dets['bit']:>10.3f} {dets['all']:>10.3f} "
              f"{dets['amp']:>10.3f}{marker}")

    # ── Анализ
    print(f"\n{'=' * 82}")
    print("АНАЛИЗ")
    print(f"{'=' * 82}")

    print(f"\n  Влияние амплитуды:")
    print(f"    FAB:    bit-only AUC = {results['FAB']['bit']:.4f}, "
          f"bit+amp = {results['FAB']['all']:.4f}")
    print(f"    Square: bit-only AUC = {results['Square']['bit']:.4f}, "
          f"bit+amp = {results['Square']['all']:.4f}")
    print(f"    APGD-CE: bit-only AUC = {results['APGD-CE']['bit']:.4f}, "
          f"bit+amp = {results['APGD-CE']['all']:.4f}")

    fab_gain = results["FAB"]["all"] - results["FAB"]["bit"]
    if fab_gain > 0.20:
        print(f"\n  ✓✓✓ Амплитуда решает проблему FAB:")
        print(f"      Bit-only:  {results['FAB']['bit']:.4f}")
        print(f"      Bit+amp:   {results['FAB']['all']:.4f}")
        print(f"      Δ = +{fab_gain:.4f}")
        print(f"\n  Публикационное утверждение:")
        print(f"      «Bit-plane signature имеет порог чувствительности")
        print(f"       по амплитуде. Добавление амплитудных признаков")
        print(f"       (L2, L∞, L0) решает проблему.»")

    # ── Сохранение
    with open("combined_features_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "asr", "auc_bit", "auc_bit_amp", "auc_amp"])
        for name in attacks:
            r = results[name]
            w.writerow([name, f"{attacks[name]['asr']:.4f}",
                        f"{r['bit']:.4f}", f"{r['all']:.4f}",
                        f"{r['amp']:.4f}"])
    print(f"\n  Сохранено: combined_features_results.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs, eps=args.eps)


if __name__ == "__main__":
    main()