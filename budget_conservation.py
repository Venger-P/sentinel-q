"""
budget_conservation.py — Adversarial Perturbation Budget Conservation.

Гипотеза: любая adversarial-атака оставляет след хотя бы в одном
структурном канале. Не существует атаки, невидимой для всех 6 каналов
одновременно.

6 каналов:
  1. Bit-channel    — frag, td_bit, hdiff_bit, H_bit
  2. JPEG-channel   — jpeg_residual histogram
  3. Edge-channel   — edge_density, grad
  4. Local-var      — local_var, laplacian
  5. DCT-channel    — dct coefficients
  6. Wavelet        — wav energy

6 атак:
  FGSM, APGD-CE, APGD-DLR, APGD-T, FAB, Square

Главная метрика:
  best_channel_AUC(attack) = max над каналами AUC на этой атаке
  Если > 0.85 для всех атак → conservation holds.

Плюс финальный детектор:
  "OR по каналам": proba = max(proba_channel) по 6 моделям.

Запуск:
    python budget_conservation.py --n_benign 500
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


# ── 6 каналов признаков ─────────────────────────────────────

def channel_bit(img_np):
    """Bit-plane channel."""
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
        out[f"vdiff_bit{bit}"] = float(np.abs(bp[1:, :] - bp[:-1, :]).mean())
    return out


def channel_jpeg(img_np):
    """JPEG residual channel."""
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


def channel_edge(img_np):
    """Edge density channel."""
    out = {}
    gx = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1])
    gy = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :])
    out["edge_density_h"] = float((gx > 2.0 / 255).mean())
    out["edge_density_v"] = float((gy > 2.0 / 255).mean())
    out["grad_h_mean"] = float(gx.mean())
    out["grad_v_mean"] = float(gy.mean())
    out["grad_h_max"] = float(gx.max())
    out["grad_v_max"] = float(gy.max())
    tv_h = np.abs(img_np[:, 1:, :] - img_np[:, :-1, :]).mean()
    tv_w = np.abs(img_np[:, :, 1:] - img_np[:, :, :-1]).mean()
    out["tv_h"] = float(tv_h)
    out["tv_w"] = float(tv_w)
    out["tv_total"] = float(tv_h + tv_w)
    return out


def channel_local(img_np):
    """Local variance / laplacian channel."""
    out = {}
    t = torch.from_numpy(img_np).float().unsqueeze(0)
    mean_local = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
    sq_local = F.avg_pool2d(t * t, kernel_size=3, stride=1, padding=1)
    var_local = (sq_local - mean_local ** 2).clamp(min=0)
    out["local_var_mean"] = float(var_local.mean())
    out["local_var_max"] = float(var_local.max())
    out["local_var_std"] = float(var_local.std())
    C = img_np.shape[0]
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    lap = F.conv2d(t, lap_kernel, padding=1, groups=C)
    out["laplacian_energy"] = float((lap ** 2).mean())
    out["laplacian_max"] = float(lap.abs().max())
    low = F.avg_pool2d(t, kernel_size=2, stride=2)
    low_up = F.interpolate(low, size=(32, 32), mode='nearest')
    high_freq = (t - low_up).abs()
    out["hf_mean"] = float(high_freq.mean())
    out["hf_max"] = float(high_freq.max())
    out["hf_std"] = float(high_freq.std())
    return out


def channel_dct(img_np):
    """DCT channel."""
    out = {}
    try:
        from scipy.fftpack import dct
        y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        y = (y * 255) - 128
        coeffs = []
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                block = y[i:i+8, j:j+8]
                if block.shape != (8, 8):
                    continue
                d = dct(dct(block.T, norm='ortho').T, norm='ortho')
                coeffs.append(d.flatten())
        if not coeffs:
            raise ValueError("no blocks")
        coeffs = np.array(coeffs)
        out["dct_dc_mean"] = float(np.abs(coeffs[:, 0]).mean())
        out["dct_dc_std"] = float(coeffs[:, 0].std())
        ac = coeffs[:, 1:]
        out["dct_ac_mean"] = float(np.abs(ac).mean())
        out["dct_ac_std"] = float(ac.std())
        out["dct_ac_max"] = float(np.abs(ac).max())
        low_freq = np.abs(ac[:, :8]).mean()
        high_freq = np.abs(ac[:, -8:]).mean()
        out["dct_low_freq"] = float(low_freq)
        out["dct_high_freq"] = float(high_freq)
        out["dct_hf_lf_ratio"] = float(high_freq / max(low_freq, 1e-9))
        abs_ac = np.abs(ac).flatten()
        hist, _ = np.histogram(abs_ac, bins=20, range=(0, 100))
        probs = hist / max(hist.sum(), 1)
        probs = probs[probs > 0]
        out["dct_entropy"] = float(-np.sum(probs * np.log2(probs)))
    except Exception:
        for k in ["dct_dc_mean", "dct_dc_std", "dct_ac_mean",
                   "dct_ac_std", "dct_ac_max", "dct_low_freq",
                   "dct_high_freq", "dct_hf_lf_ratio", "dct_entropy"]:
            out[k] = 0.0
    return out


def channel_wavelet(img_np):
    """Wavelet energy channel."""
    out = {}
    try:
        def haar_2d(x):
            H, W = x.shape
            if H % 2 != 0:
                x = x[:-1, :]; H -= 1
            if W % 2 != 0:
                x = x[:, :-1]; W -= 1
            LL = (x[0::2, 0::2] + x[0::2, 1::2]
                  + x[1::2, 0::2] + x[1::2, 1::2]) / 4
            LH = (x[0::2, 0::2] - x[0::2, 1::2]
                  + x[1::2, 0::2] - x[1::2, 1::2]) / 4
            HL = (x[0::2, 0::2] + x[0::2, 1::2]
                  - x[1::2, 0::2] - x[1::2, 1::2]) / 4
            HH = (x[0::2, 0::2] - x[0::2, 1::2]
                  - x[1::2, 0::2] + x[1::2, 1::2]) / 4
            return LL, LH, HL, HH

        y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        LL1, LH1, HL1, HH1 = haar_2d(y)
        out["wav_L1_LH"] = float((LH1 ** 2).mean())
        out["wav_L1_HL"] = float((HL1 ** 2).mean())
        out["wav_L1_HH"] = float((HH1 ** 2).mean())
        LL2, LH2, HL2, HH2 = haar_2d(LL1)
        out["wav_L2_LH"] = float((LH2 ** 2).mean())
        out["wav_L2_HL"] = float((HL2 ** 2).mean())
        out["wav_L2_HH"] = float((HH2 ** 2).mean())
        out["wav_L1_total"] = float(
            (LH1 ** 2).mean() + (HL1 ** 2).mean() + (HH1 ** 2).mean()
        )
        out["wav_L2_total"] = float(
            (LH2 ** 2).mean() + (HL2 ** 2).mean() + (HH2 ** 2).mean()
        )
        out["wav_ratio"] = out["wav_L1_total"] / max(
            out["wav_L2_total"], 1e-9
        )
    except Exception:
        for k in ["wav_L1_LH", "wav_L1_HL", "wav_L1_HH",
                   "wav_L2_LH", "wav_L2_HL", "wav_L2_HH",
                   "wav_L1_total", "wav_L2_total", "wav_ratio"]:
            out[k] = 0.0
    return out


CHANNELS = {
    "bit":       channel_bit,
    "JPEG":      channel_jpeg,
    "edge":      channel_edge,
    "local_var": channel_local,
    "DCT":       channel_dct,
    "wavelet":   channel_wavelet,
}


def extract_all(img_np):
    out = {}
    for name, fn in CHANNELS.items():
        for k, v in fn(img_np).items():
            out[f"{name}__{k}"] = v
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


# ── Основной эксперимент ────────────────────────────────────

def run(n_benign=500, n_workers=4, eps=0.05, target_fpr=0.05):
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from autoattack import AutoAttack

    print("=" * 84)
    print(f"Adversarial Perturbation Budget Conservation")
    print(f"Device: {DEVICE}, eps = {eps}")
    print("=" * 84)

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

    # ── Генерация 6 атак
    print(f"\n[1] Генерация атак ...")
    attacks = {}

    # FGSM (свой)
    from torch.autograd import grad as torch_grad
    x_fgsm = x_b_test.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_fgsm), y_b_test)
    g = torch_grad(loss, x_fgsm)[0]
    x_adv = (x_fgsm + eps * g.sign()).detach().clamp(0, 1)
    attacks["FGSM"] = {"test": x_adv}
    x_fgsm_train = x_b_train.clone().detach().requires_grad_(True)
    loss_t = F.cross_entropy(model(x_fgsm_train), y_b_train)
    g_t = torch_grad(loss_t, x_fgsm_train)[0]
    attacks["FGSM"]["train"] = (x_fgsm_train + eps * g_t.sign()).detach().clamp(0, 1)
    x_fgsm_calib = x_b_calib.clone().detach().requires_grad_(True)
    loss_c = F.cross_entropy(model(x_fgsm_calib), y_b_calib)
    g_c = torch_grad(loss_c, x_fgsm_calib)[0]
    attacks["FGSM"]["calib"] = (x_fgsm_calib + eps * g_c.sign()).detach().clamp(0, 1)
    print(f"    FGSM")

    # AutoAttack атаки
    for name, atk_id in [
        ("APGD-CE",  "apgd-ce"),
        ("APGD-DLR", "apgd-dlr"),
        ("APGD-T",   "apgd-t"),
        ("FAB",      "fab"),
        ("Square",   "square"),
    ]:
        adversary = AutoAttack(
            model, norm='Linf', eps=eps, version='custom',
            device=DEVICE, verbose=False, attacks_to_run=[atk_id],
        )
        t0 = time.time()
        attacks[name] = {
            "train": adversary.run_standard_evaluation(x_b_train, y_b_train, bs=100),
            "calib": adversary.run_standard_evaluation(x_b_calib, y_b_calib, bs=100),
            "test":  adversary.run_standard_evaluation(x_b_test, y_b_test, bs=100),
        }
        print(f"    {name}  ({time.time()-t0:.1f}s)")

    # ── ASR
    print(f"\n[2] ASR на test:")
    with torch.no_grad():
        pred_b = model(x_b_test).argmax(1)
        for name, d in attacks.items():
            pred_a = model(d["test"]).argmax(1)
            asr = ((pred_b != pred_a)
                    & (pred_b == y_b_test)).float().mean().item()
            print(f"    {name:<10} ASR = {asr:.3f}")

    # ── Признаки
    print(f"\n[3] Извлечение признаков (6 каналов) ...")
    t0 = time.time()
    fb_train = extract(x_b_train, n_workers)
    fb_calib = extract(x_b_calib, n_workers)
    fb_test = extract(x_b_test, n_workers)

    fa_train = {name: extract(d["train"], n_workers) for name, d in attacks.items()}
    fa_calib = {name: extract(d["calib"], n_workers) for name, d in attacks.items()}
    fa_test = {name: extract(d["test"], n_workers) for name, d in attacks.items()}
    print(f"    {time.time()-t0:.1f}s")

    all_keys = list(fb_train[0].keys())
    channel_keys = {}
    for ch in CHANNELS:
        channel_keys[ch] = [k for k in all_keys if k.startswith(f"{ch}__")]
    print(f"    Признаков: {len(all_keys)}")
    for ch, k in channel_keys.items():
        print(f"      {ch:<10}: {len(k)}")

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    y_cal = np.concatenate([np.zeros(len(fb_calib)),
                             np.ones(len(fa_calib["FGSM"]))])
    y_test = np.concatenate([np.zeros(len(fb_test)),
                              np.ones(len(fa_test["FGSM"]))])

    # ── Главная матрица
    print(f"\n[4] МАТРИЦА: attack × channel (AUC):")
    print(f"    {'attack':<10}", end="")
    for ch in CHANNELS:
        print(f" {ch:>10}", end="")
    print(f"  {'BEST':>8}")
    print("    " + "-" * (10 + 11 * len(CHANNELS) + 10))

    matrix = {}
    det_matrix = {}

    for atk_name in attacks:
        matrix[atk_name] = {}
        det_matrix[atk_name] = {}

        print(f"    {atk_name:<10}", end="")
        for ch in CHANNELS:
            kset = channel_keys[ch]
            X_tr = np.vstack([to_X(fb_train, kset),
                               to_X(fa_train[atk_name], kset)])
            y_tr = np.concatenate([np.zeros(len(fb_train)),
                                    np.ones(len(fa_train[atk_name]))])
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob_te_b = clf.predict_proba(to_X(fb_test, kset))[:, 1]
            prob_te_a = clf.predict_proba(to_X(fa_test[atk_name], kset))[:, 1]
            y_t = np.concatenate([np.zeros(len(prob_te_b)),
                                   np.ones(len(prob_te_a))])
            auc = roc_auc_score(y_t,
                                 np.concatenate([prob_te_b, prob_te_a]))
            matrix[atk_name][ch] = auc

            prob_cal_b = clf.predict_proba(to_X(fb_calib, kset))[:, 1]
            thr = threshold_for_fpr(prob_cal_b, target_fpr)
            det = float((prob_te_a > thr).mean())
            det_matrix[atk_name][ch] = det

            marker = "*" if auc > 0.85 else " "
            print(f" {auc:>9.3f}{marker}", end="")

        best = max(matrix[atk_name].values())
        print(f"  {best:>8.3f}")

    print(f"\n    (* — AUC > 0.85)")

    # ── Главная проверка: conservation law
    print(f"\n{'=' * 84}")
    print("ПРОВЕРКА ГИПОТЕЗЫ: Adversarial Perturbation Budget Conservation")
    print(f"{'=' * 84}")

    print(f"\n  Для каждой атаки: best_channel_AUC = max над каналами")
    print(f"  {'attack':<10} {'best_channel':<12} {'AUC':>8} "
          f"{'conservation':>15}")
    print("  " + "-" * 50)

    violations = []
    holds = []
    for atk_name in attacks:
        best_ch = max(matrix[atk_name], key=matrix[atk_name].get)
        best_auc = matrix[atk_name][best_ch]
        if best_auc > 0.85:
            verdict = "✓ holds"
            holds.append(atk_name)
        else:
            verdict = "✗ VIOLATED"
            violations.append(atk_name)
        print(f"  {atk_name:<10} {best_ch:<12} {best_auc:>8.3f} "
              f"{verdict:>15}")

    print(f"\n  Вывод:")
    if not violations:
        print(f"    ✓✓✓ CONSERVATION LAW CONFIRMED")
        print(f"        Все {len(holds)} атак имеют канал с AUC > 0.85.")
        print(f"        Каждая атака 'светится' хотя бы в одном канале.")
    elif len(violations) <= 1:
        print(f"    ~ Почти подтверждено. Нарушение: {violations}")
    else:
        print(f"    ✗ Conservation не подтверждён для {violations}")

    # ── Финальный детектор: OR по каналам
    print(f"\n[5] Финальный детектор: OR по каналам")
    print(f"    {'attack':<10} {'AUC_OR':>9} {'det_OR@5%':>11} "
          f"{'лучший канал':>15}")
    print("    " + "-" * 50)

    or_results = {}
    for atk_name in attacks:
        # Обучаем 6 моделей и собираем вероятности
        prob_te_b_max = np.zeros(len(fb_test))
        prob_te_a_max = np.zeros(len(fa_test[atk_name]))
        prob_cal_b_max = np.zeros(len(fb_calib))

        for ch in CHANNELS:
            kset = channel_keys[ch]
            X_tr = np.vstack([to_X(fb_train, kset),
                               to_X(fa_train[atk_name], kset)])
            y_tr = np.concatenate([np.zeros(len(fb_train)),
                                    np.ones(len(fa_train[atk_name]))])
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob_te_b_max = np.maximum(prob_te_b_max,
                clf.predict_proba(to_X(fb_test, kset))[:, 1])
            prob_te_a_max = np.maximum(prob_te_a_max,
                clf.predict_proba(to_X(fa_test[atk_name], kset))[:, 1])
            prob_cal_b_max = np.maximum(prob_cal_b_max,
                clf.predict_proba(to_X(fb_calib, kset))[:, 1])

        y_t = np.concatenate([np.zeros(len(prob_te_b_max)),
                               np.ones(len(prob_te_a_max))])
        auc_or = roc_auc_score(y_t,
                                np.concatenate([prob_te_b_max,
                                                prob_te_a_max]))
        thr = threshold_for_fpr(prob_cal_b_max, target_fpr)
        det_or = float((prob_te_a_max > thr).mean())

        best_ch = max(matrix[atk_name], key=matrix[atk_name].get)
        or_results[atk_name] = {"auc": auc_or, "det": det_or}

        marker = ""
        if det_or >= 0.85:
            marker = " ✓✓✓"
        elif det_or >= 0.70:
            marker = " ✓✓"

        print(f"    {atk_name:<10} {auc_or:>9.4f} {det_or:>11.3f} "
              f"{best_ch:>15}{marker}")

    # ── Сохранение
    with open("budget_conservation_matrix.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack"] + list(CHANNELS) + ["best", "best_channel"])
        for atk_name in attacks:
            best_ch = max(matrix[atk_name], key=matrix[atk_name].get)
            best_auc = matrix[atk_name][best_ch]
            w.writerow([atk_name]
                        + [f"{matrix[atk_name][ch]:.4f}" for ch in CHANNELS]
                        + [f"{best_auc:.4f}", best_ch])

    with open("budget_conservation_or.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "auc_or", "det_or_at_5pct"])
        for atk_name, r in or_results.items():
            w.writerow([atk_name, f"{r['auc']:.4f}",
                        f"{r['det']:.4f}"])

    print(f"\n  Сохранено: budget_conservation_matrix.csv")
    print(f"  Сохранено: budget_conservation_or.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_benign", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()
    run(n_benign=args.n_benign, n_workers=args.jobs,
        eps=args.eps, target_fpr=args.target_fpr)


if __name__ == "__main__":
    main()