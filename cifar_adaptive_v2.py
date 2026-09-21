"""
cifar_adaptive_v2.py — правильный протокол adaptive attack на CIFAR-10.

Исправление: детектор обучается на ТОЙ ЖЕ атаке, которую потом
адаптируем (PGD-20), а не на FGSM из кэша.

Протокол:
  1. Загрузить benign (1000 примеров)
  2. Сгенерировать PGD-20 (eps=0.05) — 1000 примеров
  3. Train/test split (80/20)
  4. Обучить LR на train (benign + PGD)
  5. Замерить FPR и detection на test
  6. Adaptive sweep на test (регенерация PGD с штрафом)

Запуск:
    python cifar_adaptive_v2.py --n 500
"""

import argparse
import csv
import pickle
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Модель ──────────────────────────────────────────────────

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


# ── Bit-признаки (те же, что в bit_features.py) ─────────────

SBF_KEYS = [
    "td_bit5", "hdiff_bit5", "td_bit4", "td_bit6", "hdiff_bit4",
    "frag_bit4", "td_bit0", "frag_bit3", "td_bit2", "H_bit5",
]


def real_bit_features(img_np):
    out = {}
    arr = (img_np * 255).astype(np.uint8)
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
    return {k: out[k] for k in SBF_KEYS if k in out}


# ── Дифференцируемые bit-признаки для adaptive PGD ────────

def soft_transition_density(x, bit):
    N = x.size(0)
    x255 = x.reshape(N, -1) * 255.0
    scale = 2 ** bit
    bv = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)
    bv_s = torch.cat([bv[:, 1:], bv[:, :1]], dim=1)
    return ((bv - bv_s) ** 2).mean(dim=1)


def soft_hdiff(x, bit):
    x255 = x * 255.0
    scale = 2 ** bit
    bv = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)
    return ((bv[:, :, :, 1:] - bv[:, :, :, :-1]) ** 2).mean(dim=(1, 2, 3))


def soft_H(x, bit):
    x255 = x * 255.0
    scale = 2 ** bit
    bv = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)
    H = -(bv * (bv + 1e-9).log()
          + (1 - bv) * (1 - bv + 1e-9).log())
    return H.mean(dim=(1, 2, 3))


def soft_frag_proxy(x, bit, p=0.1):
    x255 = x * 255.0
    scale = 2 ** bit
    bv = 0.5 + 0.5 * torch.cos(np.pi * 2.0 * x255 / scale)
    H_o = -(bv * (bv + 1e-9).log()
            + (1 - bv) * (1 - bv + 1e-9).log()).mean(dim=(1, 2, 3))
    noise = torch.randn_like(bv) * p
    bv_n = (bv + noise).clamp(0, 1)
    H_n = -(bv_n * (bv_n + 1e-9).log()
            + (1 - bv_n) * (1 - bv_n + 1e-9).log()).mean(dim=(1, 2, 3))
    return H_n - H_o


def compute_soft_features(x):
    """10 признаков в том же порядке, что SBF_KEYS."""
    return torch.stack([
        soft_transition_density(x, 5),
        soft_hdiff(x, 5),
        soft_transition_density(x, 4),
        soft_transition_density(x, 6),
        soft_hdiff(x, 4),
        soft_frag_proxy(x, 4),
        soft_transition_density(x, 0),
        soft_frag_proxy(x, 3),
        soft_transition_density(x, 2),
        soft_H(x, 5),
    ], dim=1)


# ── PGD ─────────────────────────────────────────────────────

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


def adaptive_pgd(model, x, y, x_orig, lambda_bit,
                  eps=0.05, alpha=None, n_iter=20):
    if alpha is None:
        alpha = eps / 4
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = (x + delta).clamp(0, 1).detach()
    with torch.no_grad():
        feat_orig = compute_soft_features(x_orig)
    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        ce = F.cross_entropy(model(x_adv), y)
        if lambda_bit > 0:
            feat = compute_soft_features(x_adv)
            penalty = (feat - feat_orig).abs().mean()
            loss = ce + lambda_bit * penalty
        else:
            loss = ce
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
    model = SmallCNN32().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, x, y


# ── Основной эксперимент ────────────────────────────────────

def run(n_samples=500, eps=0.05, n_iter=20,
        lambdas=(0.0, 1.0, 5.0, 20.0, 50.0, 200.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score

    print("=" * 78)
    print(f"Adaptive PGD — CIFAR-10 (правильный протокол)")
    print(f"Device: {DEVICE}")
    print("=" * 78)

    model, x_all, y_all = load_all()
    n = min(n_samples, len(x_all))
    x_b = x_all[:n].to(DEVICE)
    y = y_all[:n].to(DEVICE)
    print(f"\nBenign: {n} примеров, eps = {eps}, n_iter = {n_iter}")

    # ── Train/test split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(0.7 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    x_train = x_b[train_idx]
    y_train = y[train_idx]
    x_test = x_b[test_idx]
    y_test = y[test_idx]
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")

    # ── Генерация PGD на train
    print(f"\n[1] Генерация PGD-20 на train ...")
    t0 = time.time()
    x_train_adv = pgd_attack(model, x_train, y_train,
                              eps=eps, n_iter=n_iter)
    print(f"    {time.time()-t0:.1f}s")

    # Проверка ASR
    with torch.no_grad():
        pred_orig = model(x_train).argmax(1)
        pred_adv = model(x_train_adv).argmax(1)
    asr_train = (pred_orig != pred_adv).float().mean().item()
    print(f"    ASR train: {asr_train:.3f}")

    # ── Обучение LR на PGD-примерах
    print(f"\n[2] Обучение LR на train (benign + PGD-20) ...")
    feats_b = [real_bit_features(x_train[i].cpu().numpy())
               for i in range(len(x_train))]
    feats_a = [real_bit_features(x_train_adv[i].cpu().numpy())
               for i in range(len(x_train_adv))]

    X_train = np.array(
        [[f[k] for k in SBF_KEYS] for f in feats_b + feats_a]
    )
    y_train_lr = np.concatenate([np.zeros(len(feats_b)),
                                  np.ones(len(feats_a))])

    pipe = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    pipe.fit(X_train, y_train_lr)

    # ── Проверка на test
    print(f"\n[3] Проверка на test (baseline, PGD без штрафа):")
    x_test_adv = pgd_attack(model, x_test, y_test,
                             eps=eps, n_iter=n_iter)

    with torch.no_grad():
        pred_o = model(x_test).argmax(1)
        pred_a = model(x_test_adv).argmax(1)
    asr_test = ((pred_o != pred_a)
                & (pred_o == y_test)).float().mean().item()

    feats_test_b = [real_bit_features(x_test[i].cpu().numpy())
                    for i in range(len(x_test))]
    feats_test_a = [real_bit_features(x_test_adv[i].cpu().numpy())
                    for i in range(len(x_test_adv))]

    X_test_b = np.array([[f[k] for k in SBF_KEYS] for f in feats_test_b])
    X_test_a = np.array([[f[k] for k in SBF_KEYS] for f in feats_test_a])

    prob_b = pipe.predict_proba(X_test_b)[:, 1]
    prob_a = pipe.predict_proba(X_test_a)[:, 1]
    fpr = float((prob_b > 0.5).mean())
    det = float((prob_a > 0.5).mean())
    auc_test = roc_auc_score(
        np.concatenate([np.zeros(len(prob_b)), np.ones(len(prob_a))]),
        np.concatenate([prob_b, prob_a]),
    )

    print(f"    ASR test (PGD-20, λ=0):   {asr_test:.3f}")
    print(f"    FPR на benign:            {fpr:.3f}")
    print(f"    Detection на PGD-20:      {det:.3f}")
    print(f"    AUC test:                 {auc_test:.3f}")

    # ── Adaptive sweep
    print(f"\n[4] Adaptive PGD sweep:")
    print(f"    {'λ':>6} {'ASR':>8} {'detect':>9} {'trade-off':>12}")
    print("    " + "-" * 42)

    results = []
    for lam in lambdas:
        x_adv = adaptive_pgd(model, x_test, y_test, x_test,
                              lambda_bit=lam, eps=eps, n_iter=n_iter)

        with torch.no_grad():
            pred_a = model(x_adv).argmax(1)
        asr = ((pred_o != pred_a) & (pred_o == y_test)).float().mean().item()

        feats = [real_bit_features(x_adv[i].cpu().numpy())
                 for i in range(len(x_adv))]
        X_a = np.array([[f[k] for k in SBF_KEYS] for f in feats])
        prob = pipe.predict_proba(X_a)[:, 1]
        det_rate = float((prob > 0.5).mean())

        tradeoff = asr * (1 - det_rate)
        results.append({
            "lambda": lam, "asr": asr, "detection": det_rate,
            "tradeoff": tradeoff,
        })
        print(f"    {lam:>6.1f} {asr:>8.3f} {det_rate:>9.3f} "
              f"{tradeoff:>12.3f}")

    # ── Анализ
    print(f"\n{'=' * 78}")
    print("ИТОГ")
    print(f"{'=' * 78}")

    dangerous = [r for r in results
                 if r["asr"] > 0.5 and r["detection"] < 0.5]
    safe = [r for r in results
            if r["asr"] > 0.5 and r["detection"] > 0.7]

    if dangerous:
        print(f"\n✗ УЯЗВИМ к adaptive PGD")
        for r in dangerous:
            print(f"   λ={r['lambda']:.1f}: "
                  f"ASR={r['asr']:.3f}, det={r['detection']:.3f}")
    elif safe:
        print(f"\n✓ УСТОЙЧИВ к adaptive PGD")
        print(f"   На всём фронте сильная атака детектируется")
    else:
        print(f"\n~ Промежуточный результат")

    # Сохранение
    with open("cifar_adaptive_v2_results.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nСохранено: cifar_adaptive_v2_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--iter", type=int, default=20)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 1.0, 5.0, 20.0, 50.0, 200.0])
    args = ap.parse_args()
    run(n_samples=args.n, eps=args.eps, n_iter=args.iter,
        lambdas=tuple(args.lambdas))