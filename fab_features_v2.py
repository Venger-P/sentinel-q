"""
fab_features_v2.py — расширенный набор признаков для FAB.

Дополнительно к bit + abs:
  - JPEG residual (PIL, quality=85)
  - DCT коэффициенты (scipy.fftpack, 8x8 блоки)
  - LBP (Local Binary Patterns, реализация вручную)
  - Wavelet energy (Haar, 3 уровня)

Проверяем, какие из них реально помогают для FAB.

Запуск:
    python fab_features_v2.py --n_benign 500
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


# ── Базовые признаки ────────────────────────────────────────

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

    low = F.avg_pool2d(t, kernel_size=2, stride=2)
    low_up = F.interpolate(low, size=(32, 32), mode='nearest')
    high_freq = (t - low_up).abs()
    out["hf_mean"] = float(high_freq.mean())
    out["hf_max"] = float(high_freq.max())
    low_energy = float((low_up ** 2).mean())
    high_energy = float((high_freq ** 2).mean())
    out["hf_lf_ratio"] = high_energy / max(low_energy, 1e-9)
    return out


# ── НОВЫЕ признаки ──────────────────────────────────────────

def jpeg_residual_features(img_np):
    """
    JPEG residual: сжимаем с quality=85, разница с оригиналом.
    FAB может разрушить JPEG-структуру.
    """
    out = {}
    try:
        # Переводим в PIL
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

        # Гистограмма остатков (8 бинов)
        hist, _ = np.histogram(np.abs(residual).flatten(),
                                bins=8, range=(0, 0.1))
        hist = hist / max(hist.sum(), 1)
        for i, v in enumerate(hist):
            out[f"jpeg_res_hist{i}"] = float(v)

        # Энтропия остатков
        probs = hist[hist > 0]
        out["jpeg_res_entropy"] = float(-np.sum(probs * np.log2(probs)))
    except Exception:
        # Fallback если PIL не работает
        out["jpeg_res_mean"] = 0.0
        out["jpeg_res_std"] = 0.0
        out["jpeg_res_max"] = 0.0
        out["jpeg_res_energy"] = 0.0
        for i in range(8):
            out[f"jpeg_res_hist{i}"] = 0.0
        out["jpeg_res_entropy"] = 0.0
    return out


def dct_features(img_np):
    """
    DCT коэффициенты по 8x8 блокам.
    Статистика: энергия DC, средняя энергия AC, распределение.
    """
    out = {}
    try:
        from scipy.fftpack import dct

        # Работаем с Y (яркостью)
        y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        y = (y * 255) - 128  # центрируем

        # 8x8 блоки: 4x4 = 16 блоков
        dct_coeffs = []
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                block = y[i:i+8, j:j+8]
                if block.shape != (8, 8):
                    continue
                # 2D DCT
                d = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.append(d.flatten())

        if not dct_coeffs:
            raise ValueError("no blocks")

        dct_coeffs = np.array(dct_coeffs)  # (16, 64)

        # DC компонента (первый коэффициент)
        out["dct_dc_mean"] = float(np.abs(dct_coeffs[:, 0]).mean())
        out["dct_dc_std"] = float(dct_coeffs[:, 0].std())

        # AC компоненты (индексы 1..63)
        ac = dct_coeffs[:, 1:]
        out["dct_ac_mean"] = float(np.abs(ac).mean())
        out["dct_ac_std"] = float(ac.std())
        out["dct_ac_max"] = float(np.abs(ac).max())

        # Энергия в низких / высоких частотах
        # Низкие: индексы 1..8 (первые 8 AC)
        # Высокие: индексы 56..63 (последние 8 AC)
        low_freq = np.abs(ac[:, :8]).mean()
        high_freq = np.abs(ac[:, -8:]).mean()
        out["dct_low_freq"] = float(low_freq)
        out["dct_high_freq"] = float(high_freq)
        out["dct_hf_lf_ratio"] = float(high_freq / max(low_freq, 1e-9))

        # Энтропия DCT коэффициентов
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


def lbp_features(img_np):
    """
    Local Binary Patterns (8 соседей, uniform).
    Гистограмма из 10 бинов.
    """
    out = {}
    try:
        # Y канал
        y = (0.299 * img_np[0] + 0.587 * img_np[1]
              + 0.114 * img_np[2])
        y = (y * 255).astype(np.uint8)

        # LBP вручную
        h, w = y.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # 8 соседей: (dy, dx)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                      (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

        center = y[1:-1, 1:-1]

        for k, (dy, dx) in enumerate(neighbors):
            neighbor = y[1+dy:h-1+dy, 1+dx:w-1+dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << k)

        # Uniform LBP: число переходов 0→1 и 1→0
        def count_transitions(x):
            # Побитовые сдвиги
            bits = np.array([(x >> i) & 1 for i in range(8)])
            # Кольцо: сравниваем соседние + wrap
            trans = ((bits != np.roll(bits, 1))).sum()
            return trans

        # Упрощённо: гистограмма по значениям
        hist, _ = np.histogram(lbp.flatten(), bins=10,
                                range=(0, 256))
        hist = hist / max(hist.sum(), 1)

        for i, v in enumerate(hist):
            out[f"lbp_hist{i}"] = float(v)

        out["lbp_mean"] = float(lbp.mean())
        out["lbp_std"] = float(lbp.std())
    except Exception:
        for i in range(10):
            out[f"lbp_hist{i}"] = 0.0
        out["lbp_mean"] = 0.0
        out["lbp_std"] = 0.0
    return out


def wavelet_features(img_np):
    """
    Haar wavelet, 2 уровня. Энергия по поддиапазонам.
    """
    out = {}
    try:
        def haar_2d(x):
            """Один уровень Haar wavelet."""
            # x: (H, W)
            H, W = x.shape
            if H % 2 != 0:
                x = x[:-1, :]
                H -= 1
            if W % 2 != 0:
                x = x[:, :-1]
                W -= 1

            # Approximate (LL)
            LL = (x[0::2, 0::2] + x[0::2, 1::2]
                  + x[1::2, 0::2] + x[1::2, 1::2]) / 4

            # Horizontal detail (LH)
            LH = (x[0::2, 0::2] - x[0::2, 1::2]
                  + x[1::2, 0::2] - x[1::2, 1::2]) / 4

            # Vertical detail (HL)
            HL = (x[0::2, 0::2] + x[0::2, 1::2]
                  - x[1::2, 0::2] - x[1::2, 1::2]) / 4

            # Diagonal detail (HH)
            HH = (x[0::2, 0::2] - x[0::2, 1::2]
                  - x[1::2, 0::2] + x[1::2, 1::2]) / 4

            return LL, LH, HL, HH

        # Y канал
        y = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]

        # Уровень 1
        LL1, LH1, HL1, HH1 = haar_2d(y)
        out["wav_L1_LH_energy"] = float((LH1 ** 2).mean())
        out["wav_L1_HL_energy"] = float((HL1 ** 2).mean())
        out["wav_L1_HH_energy"] = float((HH1 ** 2).mean())

        # Уровень 2
        LL2, LH2, HL2, HH2 = haar_2d(LL1)
        out["wav_L2_LH_energy"] = float((LH2 ** 2).mean())
        out["wav_L2_HL_energy"] = float((HL2 ** 2).mean())
        out["wav_L2_HH_energy"] = float((HH2 ** 2).mean())

        # Отношения
        out["wav_L1_total"] = float(
            (LH1 ** 2).mean() + (HL1 ** 2).mean() + (HH1 ** 2).mean()
        )
        out["wav_L2_total"] = float(
            (LH2 ** 2).mean() + (HL2 ** 2).mean() + (HH2 ** 2).mean()
        )
        out["wav_ratio_L1_L2"] = out["wav_L1_total"] / max(
            out["wav_L2_total"], 1e-9
        )
    except Exception:
        for k in ["wav_L1_LH_energy", "wav_L1_HL_energy",
                   "wav_L1_HH_energy", "wav_L2_LH_energy",
                   "wav_L2_HL_energy", "wav_L2_HH_energy",
                   "wav_L1_total", "wav_L2_total", "wav_ratio_L1_L2"]:
            out[k] = 0.0
    return out


def extract_all_features(img_np):
    """Все признаки: bit + abs + jpeg + dct + lbp + wavelet."""
    out = {}
    out.update(bit_features(img_np))
    out.update(abs_features(img_np))
    out.update(jpeg_residual_features(img_np))
    out.update(dct_features(img_np))
    out.update(lbp_features(img_np))
    out.update(wavelet_features(img_np))
    return out


def extract(x_tensor, n_workers=4):
    arr = x_tensor.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(extract_all_features,
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

    print("=" * 82)
    print(f"FAB: расширенный набор признаков")
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

    # FAB
    print(f"\n[1] Генерация FAB ...")
    adversary = AutoAttack(
        model, norm='Linf', eps=eps, version='custom',
        device=DEVICE, verbose=False,
        attacks_to_run=["fab"],
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

    # Признаки
    print(f"\n[2] Извлечение признаков (bit + abs + jpeg + dct + lbp + wav) ...")
    t0 = time.time()
    fb_train = extract(x_b_train, n_workers)
    fb_calib = extract(x_b_calib, n_workers)
    fb_test = extract(x_b_test, n_workers)
    fa_train = extract(x_a_train, n_workers)
    fa_calib = extract(x_a_calib, n_workers)
    fa_test = extract(x_a_test, n_workers)
    print(f"    {time.time()-t0:.1f}s")

    keys = list(fb_train[0].keys())

    # Группы признаков
    bit_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("frag", "H_bit", "td_bit", "hdiff_bit", "vdiff_bit", "nu")
    ) and k != "H"]
    abs_keys = [k for k in keys if any(
        k.startswith(p) for p in
        ("tv_", "grad_", "edge_", "local_var", "laplacian_", "hf_")
    )]
    jpeg_keys = [k for k in keys if k.startswith("jpeg_")]
    dct_keys = [k for k in keys if k.startswith("dct_")]
    lbp_keys = [k for k in keys if k.startswith("lbp_")]
    wav_keys = [k for k in keys if k.startswith("wav_")]

    print(f"\n    Группы признаков:")
    print(f"      Bit:     {len(bit_keys)}")
    print(f"      Abs:     {len(abs_keys)}")
    print(f"      JPEG:    {len(jpeg_keys)}")
    print(f"      DCT:     {len(dct_keys)}")
    print(f"      LBP:     {len(lbp_keys)}")
    print(f"      Wav:     {len(wav_keys)}")
    print(f"      Всего:   {len(keys)}")

    def to_X(feats, kset):
        return np.array([[f[k] for k in kset] for f in feats])

    # Готовим комбинации
    combos = {
        "bit-only":         bit_keys,
        "abs-only":         abs_keys,
        "bit+abs":          bit_keys + abs_keys,
        "bit+abs+jpeg":     bit_keys + abs_keys + jpeg_keys,
        "bit+abs+dct":      bit_keys + abs_keys + dct_keys,
        "bit+abs+lbp":      bit_keys + abs_keys + lbp_keys,
        "bit+abs+wav":      bit_keys + abs_keys + wav_keys,
        "all":              keys,
    }

    print(f"\n[3] Сравнение наборов:")
    print(f"    {'combo':<20} {'n_feat':>7} {'AUC':>8} "
          f"{'det@1%':>8} {'det@5%':>8} {'det@10%':>9}")
    print("    " + "-" * 66)

    y_cal = np.concatenate([np.zeros(len(fb_calib)),
                             np.ones(len(fa_calib))])
    y_test = np.concatenate([np.zeros(len(fb_test)),
                              np.ones(len(fa_test))])

    results = {}
    for name, kset in combos.items():
        if not kset:
            continue

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

        thr1 = threshold_for_fpr(prob_cal_b, 0.01)
        thr5 = threshold_for_fpr(prob_cal_b, 0.05)
        thr10 = threshold_for_fpr(prob_cal_b, 0.10)

        det1 = float((prob_te_a > thr1).mean())
        det5 = float((prob_te_a > thr5).mean())
        det10 = float((prob_te_a > thr10).mean())

        results[name] = {
            "auc": auc, "det1": det1, "det5": det5, "det10": det10,
            "n_feat": len(kset),
        }

        marker = ""
        if det5 >= 0.85:
            marker = " ✓✓✓"
        elif det5 >= 0.75:
            marker = " ✓✓"
        elif det5 >= 0.65:
            marker = " ✓"

        print(f"    {name:<20} {len(kset):>7} {auc:>8.4f} "
              f"{det1:>8.3f} {det5:>8.3f} {det10:>9.3f}{marker}")

    # ── Feature importance для all
    print(f"\n[4] Top-20 признаков (all):")
    X_tr_all = np.vstack([to_X(fb_train, keys),
                           to_X(fa_train, keys)])
    y_tr_all = np.concatenate([np.zeros(len(fb_train)),
                                np.ones(len(fa_train))])
    clf_all = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf_all.fit(X_tr_all, y_tr_all)

    importances = clf_all.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for i in range(20):
        idx = sorted_idx[i]
        key = keys[idx]
        # Группа
        if key in bit_keys: kind = "BIT"
        elif key in abs_keys: kind = "ABS"
        elif key in jpeg_keys: kind = "JPG"
        elif key in dct_keys: kind = "DCT"
        elif key in lbp_keys: kind = "LBP"
        elif key in wav_keys: kind = "WAV"
        else: kind = "?"

        print(f"    {i+1:>2}. [{kind}] {key:<22} "
              f"{importances[idx]:.4f}")

    # ── Итог
    print(f"\n{'=' * 82}")
    print("ИТОГ")
    print(f"{'=' * 82}")

    best = max(results.items(), key=lambda kv: kv[1]["det5"])
    print(f"\n  Best by det@5%: {best[0]}")
    print(f"    AUC = {best[1]['auc']:.4f}")
    print(f"    det@1%  = {best[1]['det1']:.3f}")
    print(f"    det@5%  = {best[1]['det5']:.3f}")
    print(f"    det@10% = {best[1]['det10']:.3f}")
    print(f"    n_feat  = {best[1]['n_feat']}")

    print(f"\n  Прогрессия FAB:")
    print(f"    bit-only:              det@5% = "
          f"{results['bit-only']['det5']:.3f}")
    print(f"    bit+abs:               det@5% = "
          f"{results['bit+abs']['det5']:.3f}")
    print(f"    bit+abs+new:           det@5% = "
          f"{best[1]['det5']:.3f}")

    if best[1]["det5"] >= 0.85:
        print(f"\n  ✓✓✓ FAB ЗАКРЫТ полностью")
    elif best[1]["det5"] >= 0.75:
        print(f"\n  ✓✓ FAB закрыт на production уровне")
    elif best[1]["det5"] >= 0.65:
        print(f"\n  ✓ Существенное улучшение")

    # Сохранение
    with open("fab_features_v2_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "n_features", "auc", "det1", "det5", "det10"])
        for name, r in results.items():
            w.writerow([name, r["n_feat"], f"{r['auc']:.4f}",
                        f"{r['det1']:.4f}", f"{r['det5']:.4f}",
                        f"{r['det10']:.4f}"])
    print(f"\n  Сохранено: fab_features_v2_results.csv")


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