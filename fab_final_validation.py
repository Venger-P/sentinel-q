"""
fab_final_validation.py — финальная валидация через cross-validation.

Проблема: split train=300/calib=100/test=100 даёт шумные метрики.
det@1% и det@5% скачут от запуска к запуску.

Решение: 5 случайных split'ов на одних и тех же данных.
Усредняем метрики + считаем std.

Финальные числа для статьи:
  - jpeg-only (13 признаков)
  - jpeg+bit+abs (66 признаков)

Запуск:
    python fab_final_validation.py --n_benign 500
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
    out["H"] = shannon_entropy(full)
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
    return out


def abs_features(img_np):
    out = {}
    gx = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1])
    gy = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
    out["edge_density_h"] = float((gx > 2.0 / 255).mean())
    out["edge_density_v"] = float((gy > 2.0 / 255).mean())
    out["grad_h_mean"] = float(gx.mean())
    out["grad_v_mean"] = float(gy.mean())

    t = torch.from_numpy(img_np).float().unsqueeze(0)
    mean_local = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    sq_local = F.avg_pool2d(t * t, kernel_size=3, stride=1, padding=1)
    var_local = (sq_local - mean_local ** 2).clamp(min=0)
    out["local_var_mean"] = float(var_local.mean())
    out["local_var_max"] = float(var_local.max())

    tv_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :]).mean()
    tv_w = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1]).mean()
    out["tv_total"] = float(tv_h + tv_w)
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


def run(n_benign=500, n_workers=4, eps=0.05, seeds=(42, 123, 2024, 7, 13)):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 82)
    print(f"FAB: cross-validation валидация")
    print(f"Device: {DEVICE}, eps = {eps}, n_seeds = {len(seeds)}")
    print("=" * 82)

    model, x_all, y_all = load_all()
    total = len(x_all)
    n = min(n_benign, total)

    x_b = x_all[:n].to(DEVICE)
    y_b = y_all[:n].to(DEVICE)

    print(f"\n[0] Данные: {n} benign")

    # Генерируем FAB один раз на всём наборе
    print(f"\n[1] Генерация FAB ...")
    adversary = AutoAttack(
        model, norm='Linf', eps=eps, version='custom',
        device=DEVICE, verbose=False, attacks_to_run=["fab"],
    )
    t0 = time.time()
    x_a = adversary.run_standard_evaluation(x_b, y_b, bs=100)
    print(f"    {time.time()-t0:.1f}s")

    with torch.no_grad():
        pred_b = model(x_b).argmax(1)
        pred_a = model(x_a).argmax(1)
        asr = ((pred_b != pred_a) & (pred_b == y_b)).float().mean().item()
    print(f"    ASR = {asr:.3f}")

    # Извлекаем признаки один раз
    print(f"\n[2] Признаки ...")
    t0 = time.time()
    fb = extract(x_b, n_workers)
    fa = extract(x_a, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb[0].keys())
    jpeg_keys = [k for k in keys if k.startswith("jpeg_")]
    bit_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("frag", "H_bit", "td_bit", "hdiff_bit", "vdiff_bit")
    ) and k != "H"]
    abs_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("tv_", "grad_", "edge_", "local_var", "laplacian_")
    )]

    combos = {
        "jpeg-only":       jpeg_keys,
        "jpeg+bit":        jpeg_keys + bit_keys,
        "jpeg+bit+abs":    jpeg_keys + bit_keys + abs_keys,
    }

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    print(f"\n[3] {len(seeds)} split × {len(combos)} конфигураций ...")

    # Для каждой конфигурации и каждого seed — обучение и оценка
    results = {name: {"auc": [], "det1": [], "det5": [], "det10": []}
                for name in combos}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_train = int(0.6 * n)
        n_calib = int(0.2 * n)
        n_test = n - n_train - n_calib

        idx_train = perm[:n_train]
        idx_calib = perm[n_train:n_train + n_calib]
        idx_test = perm[n_train + n_calib:]

        fb_train = [fb[i] for i in idx_train]
        fb_calib = [fb[i] for i in idx_calib]
        fb_test = [fb[i] for i in idx_test]
        fa_train = [fa[i] for i in idx_train]
        fa_calib = [fa[i] for i in idx_calib]
        fa_test = [fa[i] for i in idx_test]

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

            y_test = np.concatenate([np.zeros(len(prob_te_b)),
                                      np.ones(len(prob_te_a))])
            auc = roc_auc_score(y_test,
                                 np.concatenate([prob_te_b, prob_te_a]))

            dets = {}
            for fpr_t in [0.01, 0.05, 0.10]:
                thr = threshold_for_fpr(prob_cal_b, fpr_t)
                dets[fpr_t] = float((prob_te_a > thr).mean())

            results[name]["auc"].append(auc)
            results[name]["det1"].append(dets[0.01])
            results[name]["det5"].append(dets[0.05])
            results[name]["det10"].append(dets[0.10])

    # ── Финальные метрики
    print(f"\n[4] Финальные метрики (mean ± std по {len(seeds)} seeds):")
    print(f"    {'combo':<20} {'n':>4} {'AUC':>16} "
          f"{'det@1%':>16} {'det@5%':>16}")
    print("    " + "-" * 76)

    final = {}
    for name, kset in combos.items():
        r = results[name]
        auc_m = np.mean(r["auc"]); auc_s = np.std(r["auc"])
        d1_m = np.mean(r["det1"]); d1_s = np.std(r["det1"])
        d5_m = np.mean(r["det5"]); d5_s = np.std(r["det5"])
        d10_m = np.mean(r["det10"]); d10_s = np.std(r["det10"])

        final[name] = {
            "n": len(kset),
            "auc": (auc_m, auc_s),
            "det1": (d1_m, d1_s),
            "det5": (d5_m, d5_s),
            "det10": (d10_m, d10_s),
        }

        print(f"    {name:<20} {len(kset):>4} "
              f"{auc_m:.4f} ± {auc_s:.4f}  "
              f"{d1_m:.3f} ± {d1_s:.3f}  "
              f"{d5_m:.3f} ± {d5_s:.3f}")

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    best_det5 = max(final.items(),
                     key=lambda kv: kv[1]["det5"][0])
    best_auc = max(final.items(),
                    key=lambda kv: kv[1]["auc"][0])

    print(f"\n  Лучший по det@5%: {best_det5[0]}")
    print(f"    AUC = {best_det5[1]['auc'][0]:.4f} ± "
          f"{best_det5[1]['auc'][1]:.4f}")
    print(f"    det@1% = {best_det5[1]['det1'][0]:.3f} ± "
          f"{best_det5[1]['det1'][1]:.3f}")
    print(f"    det@5% = {best_det5[1]['det5'][0]:.3f} ± "
          f"{best_det5[1]['det5'][1]:.3f}")

    print(f"\n  Лучший по AUC: {best_auc[0]}")
    print(f"    AUC = {best_auc[1]['auc'][0]:.4f} ± "
          f"{best_auc[1]['auc'][1]:.4f}")

    print(f"\n  Публикационное утверждение:")
    print(f"    JPEG residual histogram (13 признаков) детектирует")
    print(f"    FAB с AUC = {final['jpeg-only']['auc'][0]:.3f} ± "
          f"{final['jpeg-only']['auc'][1]:.3f},")
    print(f"    detection = {final['jpeg-only']['det5'][0]:.1%} "
          f"при FPR = 5%.")
    print(f"    Добавление bit и abs (66 признаков) улучшает AUC до")
    print(f"    {final['jpeg+bit+abs']['auc'][0]:.3f} ± "
          f"{final['jpeg+bit+abs']['auc'][1]:.3f},")
    print(f"    но detection остаётся на уровне "
          f"{final['jpeg+bit+abs']['det5'][0]:.1%}.")

    # Сохранение
    with open("fab_final_validation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "n_features",
                    "auc_mean", "auc_std",
                    "det1_mean", "det1_std",
                    "det5_mean", "det5_std",
                    "det10_mean", "det10_std"])
        for name, r in final.items():
            w.writerow([name, r["n"],
                        f"{r['auc'][0]:.4f}", f"{r['auc'][1]:.4f}",
                        f"{r['det1'][0]:.4f}", f"{r['det1'][1]:.4f}",
                        f"{r['det5'][0]:.4f}", f"{r['det5'][1]:.4f}",
                        f"{r['det10'][0]:.4f}", f"{r['det10'][1]:.4f}"])
    print(f"\n  Сохранено: fab_final_validation.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs, eps=args.eps)


if __name__ == "__main__":
    main()