"""
cifar_adaptive.py — adaptive PGD с bit-feature penalty на CIFAR-10.

Атакующий знает о Sentinel-Q v2 и оптимизирует шум так, чтобы:
  1. Обмануть модель (CE loss)
  2. Сохранить bit-признаки близко к benign

Penalty:
    L_total = CE(x_adv, y) + λ · Σ_i |f_i(x_adv) - f_i(x_orig)|

где f_i — 10 greedy bit-признаков:
    td_bit5, hdiff_bit5, td_bit4, td_bit6, hdiff_bit4,
    frag_bit4, td_bit0, frag_bit3, td_bit2, H_bit5

Три режима:
  1. λ = 0: обычная PGD
  2. λ > 0: атака с сохранением bit-признаков
  3. Сравнение ASR vs detection на всём Pareto-фронте

Запуск:
    python cifar_adaptive.py --n 200
    python cifar_adaptive.py --n 200 --lambdas 0 1 5 20 50 200
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy
from sentinel_q.bit_features import SBG_GREEDY_FEATURES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Модель CIFAR ────────────────────────────────────────────

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


# ── Дифференцируемые bit-признаки ───────────────────────────

def soft_transition_density(x_float, bit):
    """
    td_bit = P(x_i[bit] != x_{i+1}[bit]).

    Аппроксимация: sigmoid((x_i - x_{i+1})² / τ) усреднённая.
    """
    N = x_float.size(0)
    x_flat = x_float.reshape(N, -1)
    # Нормализуем в 0..255
    x255 = x_flat * 255.0

    # Битовое значение: (x255 // 2^bit) % 2
    # Дифференцируемая аппроксимация через sin
    # bit_val ≈ 0.5 + 0.5 * cos(2π * x255 / 2^bit)
    scale = 2 ** bit
    bit_val = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)

    # Сдвиг на 1 позицию вправо по flat
    bit_val_shifted = torch.cat(
        [bit_val[:, 1:], bit_val[:, :1]], dim=1
    )
    # td = среднее |bit_i - bit_{i+1}|² — сглаживание
    diff = (bit_val - bit_val_shifted) ** 2
    return diff.mean(dim=1)


def soft_horizontal_gradient(x_float, bit):
    """
    hdiff_bit = средний |x[i,j,bit] - x[i,j+1,bit]| по горизонтали.

    x_float: (N, C, H, W). Работаем с каждым каналом.
    """
    N, C, H, W = x_float.shape
    x255 = x_float * 255.0
    scale = 2 ** bit
    bit_val = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)

    # Горизонтальная разница
    h_diff = (bit_val[:, :, :, 1:] - bit_val[:, :, :, :-1]) ** 2
    return h_diff.mean(dim=(1, 2, 3))


def soft_frag_bit_proxy(x_float, bit, n_noise=1, p=0.1):
    """
    frag_bit — прокси через разницу H(noisy) - H(orig) на битовом слое.

    Аппроксимация через soft histogram.
    """
    N = x_float.size(0)
    x255 = x_float * 255.0
    scale = 2 ** bit
    # bit_val в [0, 1]
    bit_val = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)

    # H(orig) — энтропия битового слоя
    H_orig = -(bit_val * (bit_val + 1e-9).log()
                + (1 - bit_val) * (1 - bit_val + 1e-9).log()).mean(dim=(1, 2, 3))

    # H(noisy): добавим шум к bit_val
    H_sum = 0.0
    for _ in range(n_noise):
        noise = torch.randn_like(bit_val) * p
        bit_noisy = (bit_val + noise).clamp(0, 1)
        H_n = -(bit_noisy * (bit_noisy + 1e-9).log()
                 + (1 - bit_noisy) * (1 - bit_noisy + 1e-9).log()).mean(dim=(1, 2, 3))
        H_sum = H_sum + H_n
    H_noisy = H_sum / n_noise

    return H_noisy - H_orig


def soft_H_bit(x_float, bit):
    """H_bit — энтропия битового слоя."""
    x255 = x_float * 255.0
    scale = 2 ** bit
    bit_val = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)
    # Энтропия Бернулли для каждой "монетки"
    H = -(bit_val * (bit_val + 1e-9).log()
          + (1 - bit_val) * (1 - bit_val + 1e-9).log())
    return H.mean(dim=(1, 2, 3))


def compute_soft_features(x_float):
    """
    Возвращает tensor (N, 10) со значениями 10 greedy bit-признаков.
    """
    feats = []
    feats.append(soft_transition_density(x_float, 5))    # td_bit5
    feats.append(soft_horizontal_gradient(x_float, 5))   # hdiff_bit5
    feats.append(soft_transition_density(x_float, 4))    # td_bit4
    feats.append(soft_transition_density(x_float, 6))    # td_bit6
    feats.append(soft_horizontal_gradient(x_float, 4))   # hdiff_bit4
    feats.append(soft_frag_bit_proxy(x_float, 4))        # frag_bit4
    feats.append(soft_transition_density(x_float, 0))    # td_bit0
    feats.append(soft_frag_bit_proxy(x_float, 3))        # frag_bit3
    feats.append(soft_transition_density(x_float, 2))    # td_bit2
    feats.append(soft_H_bit(x_float, 5))                 # H_bit5
    return torch.stack(feats, dim=1)


# ── Adaptive PGD ────────────────────────────────────────────

def adaptive_pgd(model, x, y, x_orig, lambda_bit,
                  eps=0.05, alpha=None, n_iter=20):
    """
    PGD: максимизируем CE + λ · ||f(x_adv) - f(x_orig)||₁.

    Атакующий хочет:
      - обмануть модель (CE растёт)
      - сохранить bit-признаки (penalty мал)
    """
    if alpha is None:
        alpha = eps / 4

    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = (x + delta).clamp(0, 1).detach()

    # Целевые bit-признаки (от оригинала)
    with torch.no_grad():
        feat_orig = compute_soft_features(x_orig)

    for it in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        ce_loss = F.cross_entropy(logits, y)

        if lambda_bit > 0:
            feat_adv = compute_soft_features(x_adv)
            # ||f_adv - f_orig||₁ среднее по признакам
            penalty = (feat_adv - feat_orig).abs().mean()
            loss = ce_loss + lambda_bit * penalty
        else:
            loss = ce_loss

        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            delta = (x_adv - x).clamp(-eps, eps)
            x_adv = (x + delta).clamp(0, 1)

    return x_adv.detach()


# ── Загрузка ────────────────────────────────────────────────

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
    if not model_path.exists():
        raise SystemExit(f"Модель не найдена в {cache}")

    model = SmallCNN32().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, x, y


# ── Реальный bit-классификатор ─────────────────────────────

def load_bit_detector():
    """Загружает обученный LR на 10 bit-признаках."""
    import pickle
    path = Path("sentinel_q_cifar_v2.pkl")
    if not path.exists():
        raise SystemExit(
            f"{path} не найден. Запустите сначала cifar_final.py"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["keys"], data["pipeline"]


def real_bit_features(img_np):
    """Реальные bit-признаки для одного изображения."""
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
            out[f"td_bit{bit}"] = float((flat[1:] != flat[:-1]).mean())
        else:
            out[f"td_bit{bit}"] = 0.0
        out[f"hdiff_bit{bit}"] = float(
            np.abs(bp[:, 1:] - bp[:, :-1]).mean()
        )
        out[f"vdiff_bit{bit}"] = float(
            np.abs(bp[1:, :] - bp[:-1, :]).mean()
        )
    return out


# ── Оценка ──────────────────────────────────────────────────

def evaluate(model, x_orig, x_adv, y, keys, pipeline):
    with torch.no_grad():
        pred_o = model(x_orig).argmax(1)
        pred_a = model(x_adv).argmax(1)

    correct = pred_o == y
    if correct.sum() == 0:
        asr = 0.0
    else:
        asr = ((pred_o != pred_a) & correct).float().mean().item()

    # Реальные bit-признаки + LR
    x_adv_np = x_adv.cpu().numpy()
    feats_a = [real_bit_features(x_adv_np[i]) for i in range(len(x_adv_np))]
    X_a = np.array([[f[k] for k in keys] for f in feats_a])
    proba = pipeline.predict_proba(X_a)[:, 1]
    detected = (proba > 0.5).sum()
    det_rate = detected / len(proba)

    return asr, float(det_rate)


# ── Основной эксперимент ────────────────────────────────────

def run(lambdas, n_samples=200, eps=0.05, n_iter=20):
    print("=" * 78)
    print(f"Adaptive PGD с bit-feature penalty — CIFAR-10")
    print(f"Device: {DEVICE}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    x_orig = x_all[:n_samples].to(DEVICE)
    y = y_all[:n_samples].to(DEVICE)
    print(f"\nДанные: {n_samples} примеров, eps = {eps}, n_iter = {n_iter}")

    keys, pipeline = load_bit_detector()
    print(f"Классификатор: {len(keys)} bit-признаков из sentinel_q_cifar_v2.pkl")

    # Базовое состояние: benign
    x_orig_np = x_orig.cpu().numpy()
    feats_b = [real_bit_features(x_orig_np[i]) for i in range(n_samples)]
    X_b = np.array([[f[k] for k in keys] for f in feats_b])
    proba_b = pipeline.predict_proba(X_b)[:, 1]
    fpr = (proba_b > 0.5).mean()
    print(f"FPR на benign: {fpr:.3f}")

    # Проверка дифференцируемости
    print(f"\n[1] Проверка дифференцируемости soft features:")
    x_test = x_orig[:5].clone().requires_grad_(True)
    feats_test = compute_soft_features(x_test)
    feats_test.sum().backward()
    grad_norm = x_test.grad.abs().mean().item()
    print(f"    Gradient norm: {grad_norm:.6f}")
    if grad_norm < 1e-8:
        print(f"    ✗ Градиент ноль — adaptive атака бессмысленна")
        return
    print(f"    ✓ Градиент ненулевой")

    # Sweep
    print(f"\n{'=' * 78}")
    print(f"{'λ':>6} {'ASR':>8} {'detect':>9} {'trade-off':>12} "
          f"{'time':>8}")
    print(f"{'=' * 78}")

    results = []
    for lam in lambdas:
        t0 = time.time()
        x_adv = adaptive_pgd(model, x_orig, y, x_orig,
                              lambda_bit=lam,
                              eps=eps, n_iter=n_iter)
        asr, det = evaluate(model, x_orig, x_adv, y, keys, pipeline)
        tradeoff = asr * (1 - det)

        results.append({
            "lambda": lam, "asr": asr, "detection": det,
            "tradeoff": tradeoff, "time": time.time() - t0,
        })

        print(f"{lam:>6.1f} {asr:>8.3f} {det:>9.3f} "
              f"{tradeoff:>12.3f} {time.time()-t0:>7.1f}s")

    # ── Анализ
    print(f"\n{'=' * 78}")
    print("АНАЛИЗ")
    print(f"{'=' * 78}")

    print(f"\nPareto-фронт:")
    for r in results:
        if r["asr"] > 0.7 and r["detection"] < 0.5:
            m = "  ← ОПАСНО"
        elif r["asr"] > 0.7 and r["detection"] > 0.7:
            m = "  ← ХОРОШО"
        elif r["asr"] < 0.3:
            m = "  ← атака слабая"
        else:
            m = ""
        print(f"  λ={r['lambda']:>6.1f}  ASR={r['asr']:.3f}  "
              f"det={r['detection']:.3f}{m}")

    # Ключевой критерий
    dangerous = [r["lambda"] for r in results
                 if r["asr"] > 0.5 and r["detection"] < 0.5]
    safe = [r["lambda"] for r in results
            if r["asr"] > 0.5 and r["detection"] > 0.7]

    print(f"\nОпасные λ (ASR > 0.5, det < 0.5): "
          f"{dangerous if dangerous else 'нет'}")
    print(f"Безопасные λ (ASR > 0.5, det > 0.7): "
          f"{safe if safe else 'нет'}")

    # ── Итог
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    if dangerous:
        print(f"\n✗ Sentinel-Q v2 УЯЗВИМ к adaptive PGD на CIFAR-10")
        print(f"   Опасные точки:")
        for lam in dangerous:
            r = next(r for r in results if r["lambda"] == lam)
            print(f"     λ={lam:.1f}: ASR={r['asr']:.3f}, "
                  f"det={r['detection']:.3f}")
    elif safe:
        print(f"\n✓ Sentinel-Q v2 УСТОЙЧИВ")
        print(f"   На всём фронте: сильная атака → детектируется")
        print(f"   Атакующий вынужден выбирать:")
        print(f"     - либо сильная атака, но детектируемая")
        print(f"     - либо слабая атака, но невидимая")
    else:
        print(f"\n~ Промежуточный результат")
        print(f"   Ни опасных, ни полностью безопасных точек")

    # Сохранение
    with open("cifar_adaptive_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nСохранено: cifar_adaptive_results.csv")

    # ── Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        asrs = [r["asr"] for r in results]
        dets = [r["detection"] for r in results]
        ax.plot(asrs, dets, "o-", color="crimson",
                linewidth=2, markersize=12)
        for r in results:
            ax.annotate(f"λ={r['lambda']:.0f}",
                        (r["asr"], r["detection"]), fontsize=10,
                        xytext=(5, 5), textcoords="offset points")
        ax.axvspan(0.5, 1.05, ymin=0.0, ymax=0.5,
                   alpha=0.15, color="red", label="danger")
        ax.axvspan(0.5, 1.05, ymin=0.7, ymax=1.0,
                   alpha=0.15, color="green", label="safe")
        ax.set_xlabel("ASR")
        ax.set_ylabel("Detection rate")
        ax.set_title("CIFAR-10: adaptive PGD")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        lams = [r["lambda"] for r in results]
        tos = [r["tradeoff"] for r in results]
        ax.plot(lams, tos, "s-", color="darkorange",
                linewidth=2, markersize=10)
        ax.axhline(0.3, color="red", linestyle="--",
                   label="danger threshold")
        ax.set_xlabel("λ")
        ax.set_ylabel("trade-off = ASR · (1 - det)")
        ax.set_title("Trade-off vs λ")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("cifar_adaptive.png", dpi=120)
        print("Сохранено: cifar_adaptive.png")
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 1.0, 5.0, 20.0, 50.0, 200.0])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iter", type=int, default=20)
    args = ap.parse_args()
    run(args.lambdas, n_samples=args.n, eps=args.eps, n_iter=args.iter)


if __name__ == "__main__":
    main()