"""
fab_jpeg_ablation.py — минимальный набор признаков для FAB.

Открытие: JPEG residual hist1 даёт 31% важности, детектирует FAB.
DCT, LBP, Wavelet не помогают.

Абляция:
  - JPEG-only       (13)
  - JPEG + bit      (56)
  - JPEG + abs      (26)
  - JPEG + bit+abs  (69) ← текущий лидер

Запуск:
    python fab_jpeg_ablation.py --n_benign 500
"""

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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


def bit_features(img_np):
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


def abs_features(img_np):
    out = {}
    tv_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :]).mean()
    tv_w = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1]).mean()
    out["tv_h"] = float(tv_h)
    out["tv_w"] = float(tv_w)
    out["tv_total"] = float(tv_h + tv_w)

    gx = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1])
    gy = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
    out["grad_h_mean"] = float(gx.mean())
    out["grad_v_mean"] = float(gy.mean())
    out["edge_density_h"] = float((gx > 2.0 / 255).mean())
    out["edge_density_v"] = float((gy > 2.0 / 255).mean())

    t = torch.from_numpy(img_np).float().unsqueeze(0)
    mean_local = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    sq_local = F.avg_pool2d(t * t, kernel_size=3, stride=1, padding=1)
    var_local = (sq_local - mean_local ** 2).clamp(min=0)
    out["local_var_mean"] = float(var_local.mean())
    out["local_var_max"] = float(var_local.max())

    C = img_np.shape[0]
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    lap = F.conv2d(t, lap_kernel, padding=1, groups=C)
    out["laplacian_energy"] = float((lap ** 2).mean())
    return out


def jpeg_features(img_np):
    out = {}
    try:
        arr = (img_np * 255).astype(np.uint8).transpose(1, 2, 0)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        buf.seek(0)
        compressed = np.array(Image.open(buf)).astype(np.float32) / 255.0
        compressed = compressed.transpose(2, 0, 1)
        residual = img_np - compressed

        out["jpeg_res_mean"] = float(np.abs(residual).mean())
        out["jpeg_res_std"] = float(residual.std())
        out["jpeg_res_max"] = float(np.abs(residual).max())
        out["jpeg_res_energy"] = float((residual ** 2).mean())

        hist, _ = np.histogram(np.abs(residual).flatten(),
                                bins=8, range=(0, 0.1))
        hist = hist / max(hist.sum(), 1)
        for i, v in enumerate(hist):
            out[f"jpeg_res_hist{i}"] = float(v)

        probs = hist[hist > 0]
        out["jpeg_res_entropy"] = float(-np.sum(probs * np.log2(probs)))
    except Exception:
        for k in ["jpeg_res_mean", "jpeg_res_std", "jpeg_res_max",
                   "jpeg_res_energy", "jpeg_res_entropy"]:
            out[k] = 0.0
        for i in range(8):
            out[f"jpeg_res_hist{i}"] = 0.0
    return out


def extract_all(img_np):
    out = {}
    out.update(bit_features(img_np))
    out.update(abs_features(img_np))
    out.update(jpeg_features(img_np))
    return out


def extract(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(extract_all,
                              [arr[i] for i in range(len(arr))]))


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
    sorted_p = np.sort(prob_benign)[::-1]
    idx = min(int(target_fpr * len(sorted_p)), len(sorted_p) - 1)
    return float(sorted_p[idx])


def run(n_benign=500, n_workers=4, eps=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 82)
    print(f"FAB: JPEG ablation")
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

    print(f"\n[1] FAB ...")
    adversary = AutoAttack(
        model, norm='Linf', eps=eps, version='custom',
        device=DEVICE, verbose=False, attacks_to_run=["fab"],
    )
    t0 = time.time()
    x_a_train = adversary.run_standard_evaluation(x_b_train, y_b_train, bs=100)
    x_a_calib = adversary.run_standard_evaluation(x_b_calib, y_b_calib, bs=100)
    x_a_test = adversary.run_standard_evaluation(x_b_test, y_b_test, bs=100)
    print(f"    {time.time()-t0:.1f}s")

    with torch.no_grad():
        pred_b = model(x_b_test).argmax(1)
        pred_a = model(x_a_test).argmax(1)
        asr = ((pred_b != pred_a) & (pred_b == y_b_test)).float().mean().item()
    print(f"    ASR = {asr:.3f}")

    print(f"\n[2] Признаки ...")
    t0 = time.time()
    fb_train = extract(x_b_train, n_workers)
    fb_calib = extract(x_b_calib, n_workers)
    fb_test = extract(x_b_test, n_workers)
    fa_train = extract(x_a_train, n_workers)
    fa_calib = extract(x_a_calib, n_workers)
    fa_test = extract(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())
    bit_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("frag", "H_bit", "td_bit", "hdiff_bit", "vdiff_bit", "nu")
    ) and k != "H"]
    abs_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("tv_", "grad_", "edge_", "local_var", "laplacian_")
    )]
    jpeg_keys = [k for k in keys if k.startswith("jpeg_")]

    combos = {
        "jpeg-only":       jpeg_keys,
        "jpeg+bit":        jpeg_keys + bit_keys,
        "jpeg+abs":        jpeg_keys + abs_keys,
        "jpeg+bit+abs":    jpeg_keys + bit_keys + abs_keys,
        "bit+abs (baseline)": bit_keys + abs_keys,
    }

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    y_test = np.concatenate([np.zeros(len(fb_test)),
                              np.ones(len(fa_test))])

    print(f"\n[3] Абляция:")
    print(f"    {'combo':<22} {'n':>4} {'AUC':>8} "
          f"{'det@1%':>8} {'det@5%':>8} {'det@10%':>9}")
    print("    " + "-" * 62)

    results = {}
    for name, kset in combos.items():
        X_tr = np.vstack([to_X(fb_train, kset),
                           to_X(fa_train, kset)])
        y_tr = np.concatenate([np.zeros(len(fb_train)),
                                np.ones(len(fa_train))])

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_tr, y_tr)

        prob_cal_b = clf.predict_proba(to_X(fb_calib, kset))[:, 1]
        prob_te_b = clf.predict_proba(to_X(fb_test, kset))[:, 1]
        prob_te_a = clf.predict_proba(to_X(fa_test, kset))[:, 1]

        auc = roc_auc_score(y_test,
                             np.concatenate([prob_te_b, prob_te_a]))

        dets = {}
        for fpr_t in [0.01, 0.05, 0.10]:
            thr = threshold_for_fpr(prob_cal_b, fpr_t)
            dets[fpr_t] = float((prob_te_a > thr).mean())

        results[name] = {
            "auc": auc, "det1": dets[0.01],
            "det5": dets[0.05], "det10": dets[0.10],
            "n": len(kset),
        }

        marker = ""
        if dets[0.05] >= 0.85:
            marker = " ✓✓✓"
        elif dets[0.05] >= 0.75:
            marker = " ✓✓"

        print(f"    {name:<22} {len(kset):>4} {auc:>8.4f} "
              f"{dets[0.01]:>8.3f} {dets[0.05]:>8.3f} "
              f"{dets[0.10]:>9.3f}{marker}")

    # Feature importance для jpeg+bit+abs
    print(f"\n[4] Top-10 jpeg+bit+abs:")
    all_keys = jpeg_keys + bit_keys + abs_keys
    X_tr_all = np.vstack([to_X(fb_train, all_keys),
                           to_X(fa_train, all_keys)])
    y_tr_all = np.concatenate([np.zeros(len(fb_train)),
                                np.ones(len(fa_train))])
    clf_all = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf_all.fit(X_tr_all, y_tr_all)
    importances = clf_all.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(10):
        idx = sorted_idx[i]
        key = all_keys[idx]
        kind = "JPG" if key in jpeg_keys else (
            "BIT" if key in bit_keys else "ABS"
        )
        print(f"    {i+1:>2}. [{kind}] {key:<22} {importances[idx]:.4f}")

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    print(f"\n  Прогрессия по минимальному набору:")
    for name, r in results.items():
        print(f"    {name:<22} n={r['n']:>3}  "
              f"AUC={r['auc']:.4f}  det@1%={r['det1']:.3f}  "
              f"det@5%={r['det5']:.3f}")

    best = max(results.items(),
                key=lambda kv: kv[1]["det1"] + kv[1]["det5"])

    print(f"\n  Лучший баланс: {best[0]}")

    # Сохранение
    with open("fab_jpeg_ablation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "n_features", "auc", "det1", "det5", "det10"])
        for name, r in results.items():
            w.writerow([name, r["n"], f"{r['auc']:.4f}",
                        f"{r['det1']:.4f}", f"{r['det5']:.4f}",
                        f"{r['det10']:.4f}"])
    print(f"\n  Сохранено: fab_jpeg_ablation.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs, eps=args.eps)


if __name__ == "__main__":
    main()