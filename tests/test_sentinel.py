"""
Unit-тесты для Sentinel-Q.

Запуск:
    pytest tests/ -v
    # или без pytest:
    python tests/test_sentinel.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import (
    shannon_entropy, n_unique_bytes, fragility, byte_stats,
    QorbDescriptor,
)
from sentinel_q.profile import ReferenceProfile
from sentinel_q.detector import SentinelDetector


# ── Хелперы ─────────────────────────────────────────────────

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    line = f"  [{status}] {name}"
    if not cond and detail:
        line += f"  — {detail}"
    print(line)
    if cond:
        PASSED.append(name)
    else:
        FAILED.append((name, detail))


def random_bytes(n, seed=0):
    return np.random.default_rng(seed).integers(
        0, 256, size=n, dtype=np.uint8
    ).tobytes()


def constant_bytes(n, value=0x42):
    return bytes([value]) * n


def periodic_bytes(period=32, repeats=64):
    return (bytes(range(period)) * repeats)


def to_samples(arrays):
    """(N, C, H, W) float [0,1] -> list of bytes."""
    arr = (arrays * 255).astype(np.uint8)
    out = []
    for i in range(len(arr)):
        if arr.ndim == 4 and arr.shape[1] == 1:
            out.append(arr[i, 0].tobytes())
        else:
            out.append(arr[i].tobytes())
    return out


# ── Тесты core ──────────────────────────────────────────────

def test_core():
    print("\n[1] core: entropy, n_unique, fragility")

    noise = random_bytes(4096, seed=1)
    h = shannon_entropy(noise)
    check("H(шум) ≈ 8.0", 7.9 < h <= 8.0, f"H={h:.4f}")

    const = constant_bytes(4096)
    check("H(константа) = 0", shannon_entropy(const) == 0.0)
    check("n_unique(константа) = 1", n_unique_bytes(const) == 1)
    check("n_unique(шум) > 200", n_unique_bytes(noise) > 200)

    frag_noise = fragility(noise)
    frag_periodic = fragility(periodic_bytes())
    check("frag(шум) < 0.05", frag_noise < 0.05, f"frag={frag_noise:.4f}")
    check("frag(периодика) > 0.1", frag_periodic > 0.1,
          f"frag={frag_periodic:.4f}")


def test_qorb_descriptor():
    print("\n[2] QorbDescriptor: fit + report")

    noise = random_bytes(4096, seed=2)
    qd = QorbDescriptor(compressors=("zlib",), n_points=8, n_trials=1, max_p=0.3)
    qd.fit(noise)
    check("w не NaN", not np.isnan(qd.w))
    check("w ≥ 5.0 для шума", qd.w >= 5.0, f"w={qd.w:.3f}")
    check("CV ≥ 0", qd.CV >= 0)
    check("report не пустой", len(qd.report()) > 0)


# ── Тесты profile ───────────────────────────────────────────

def test_profile_build_save_load():
    print("\n[3] profile: build, save, load")

    rng = np.random.default_rng(3)
    benign = rng.integers(0, 100, size=(50, 1, 28, 28)).astype(np.float32) / 100
    adv = rng.integers(0, 256, size=(50, 1, 28, 28)).astype(np.float32) / 255
    labels = ([0] * 25 + [1] * 25)
    samples = to_samples(np.vstack([benign[:25], adv[:25]]))
    samples += to_samples(np.vstack([benign[25:], adv[25:]]))

    p = ReferenceProfile()
    p.build(samples, labels + labels)
    check("stats заполнен", len(p.stats) > 0)
    check("2 класса", len(p.stats) == 2, f"classes={list(p.stats)}")
    check("n_samples_total == 100", p.n_samples_total == 100)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        p.save(path)
        check("файл создан", path.exists())

        p2 = ReferenceProfile.load(path)
        check("загружено n_samples", p2.n_samples_total == p.n_samples_total)
        check("загружено classes",
              set(p2.stats.keys()) == set(p.stats.keys()))

        # Точечная проверка
        s = p.stats[0]
        for k in ("frag_mean", "frag_std", "H_mean", "n_unique_mean"):
            check(f"ключ {k} присутствует", k in s)


def test_profile_zscores():
    print("\n[4] profile: z_score и quantile_rank")

    rng = np.random.default_rng(4)
    samples = []
    labels = []
    for _ in range(30):
        samples.append(random_bytes(1024, seed=rng.integers(0, 10**6)))
        labels.append(0)
    for _ in range(30):
        samples.append(periodic_bytes(32, 32))
        labels.append(0)

    p = ReferenceProfile().build(samples, labels)
    # benign random z должен быть около 0
    z = p.z_score(random_bytes(1024, seed=999), 0, "frag")
    check("|z| мал для in-distribution", abs(z) < 3, f"z={z:.3f}")

    qr = p.quantile_rank(random_bytes(1024, seed=999), 0, "frag")
    check("quantile в [0, 1]", 0.0 <= qr <= 1.0, f"qr={qr:.3f}")


# ── Тесты detector ──────────────────────────────────────────

def test_detector_modes():
    print("\n[5] detector: combine modes")

    from sentinel_q.detector import SentinelDetector

    rng = np.random.default_rng(5)
    # Профиль: benign = структурированные (периодика), adv = шум
    benign_samples = [periodic_bytes(32, 32) for _ in range(40)]
    labels = [0] * 40

    p = ReferenceProfile().build(benign_samples, labels)
    det = SentinelDetector(p, z_threshold=1.5, combine="weighted")
    v = det.check(random_bytes(1024, seed=6), 0)
    check("weighted: adversarial detected", v.is_adversarial,
          f"score={v.score:.3f}")

    v2 = det.check(periodic_bytes(32, 32), 0)
    check("weighted: benign не flagged", not v2.is_adversarial,
          f"score={v2.score:.3f}")

    # Все combine режимы работают
    for combine in ("weighted", "frag_only", "mean", "max"):
        d = SentinelDetector(p, z_threshold=1.5, combine=combine)
        v = d.check(random_bytes(1024, seed=7), 0)
        check(f"combine={combine}: имеет score", isinstance(v.score, float))
        check(f"combine={combine}: has reason", len(v.reason) > 0)


def test_detector_batch():
    print("\n[6] detector: batch + ROC")

    rng = np.random.default_rng(6)
    benign = [periodic_bytes(32, 32) for _ in range(50)]
    adv = [random_bytes(1024, seed=int(rng.integers(0, 10**6)))
           for _ in range(50)]

    p = ReferenceProfile().build(benign, [0] * 50)
    d = SentinelDetector(p, z_threshold=1.5, combine="weighted")

    samples = benign + adv
    labels = [0] * 100
    is_adv = [False] * 50 + [True] * 50

    try:
        from sklearn.metrics import roc_auc_score
        metrics = d.batch_roc(samples, labels, is_adv)
        check("ROC-AUC > 0.9", metrics["roc_auc"] > 0.9,
              f"AUC={metrics['roc_auc']:.3f}")
        check("Accuracy > 0.8", metrics["accuracy"] > 0.8,
              f"acc={metrics['accuracy']:.3f}")
    except ImportError:
        check("sklearn не установлен — пропуск", True)


# ── Тесты server (только если fastapi установлен) ───────────

def test_server_imports():
    print("\n[7] server: import + create_app")

    try:
        from sentinel_q.server import create_app
    except ImportError:
        check("fastapi не установлен — пропуск", True)
        return

    rng = np.random.default_rng(7)
    benign = [periodic_bytes(32, 32) for _ in range(30)]
    p = ReferenceProfile().build(benign, [0] * 30)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        p.save(path)

        app = create_app(profile_path=str(path))
        check("app создан", app is not None)
        routes = [r.path for r in app.routes]
        check("/healthz есть", "/healthz" in routes)
        check("/v1/detect есть", "/v1/detect" in routes)
        check("/v1/detect/batch есть", "/v1/detect/batch" in routes)


# ── Тесты torch_wrapper (опционально) ───────────────────────

def test_torch_wrapper():
    print("\n[8] torch_wrapper: import + basic forward")

    try:
        import torch
        import torch.nn as nn
        from sentinel_q.torch_wrapper import (
            SentinelWrapper, AdversarialDetectedError,
        )
    except ImportError:
        check("torch не установлен — пропуск", True)
        return

    class AlwaysZero(nn.Module):
        """Модель, которая всегда предсказывает класс 0.

        Нужна, потому что профиль строится только для класса 0,
        а необученный Linear давал бы случайные метки.
        """
        def forward(self, x):
            logits = torch.zeros(x.size(0), 10)
            logits[:, 0] = 10.0
            return logits

    rng = np.random.default_rng(8)

    # Benign: разреженный структурированный паттерн (как MNIST)
    x_benign_np = np.zeros((40, 1, 28, 28), dtype=np.float32)
    for i in range(40):
        col = int(rng.integers(5, 23))
        x_benign_np[i, 0, :, col] = 1.0
        x_benign_np[i, 0, :, col + 1] = 0.5

    # Adversarial: чистый шум (гарантированно отличается)
    x_adv_np = rng.random((40, 1, 28, 28)).astype(np.float32)

    # Профиль на benign
    benign_samples = []
    for i in range(40):
        arr = (x_benign_np[i, 0] * 255).astype(np.uint8)
        benign_samples.append(arr.tobytes())

    p = ReferenceProfile().build(benign_samples, [0] * 40)

    # Диагностика: frag(benign) должен быть выше frag(adv)
    from sentinel_q.core import byte_stats
    frag_b = np.mean([byte_stats(s)["frag"] for s in benign_samples])
    frag_a = np.mean([
        byte_stats((x_adv_np[i, 0] * 255).astype(np.uint8).tobytes())["frag"]
        for i in range(40)
    ])
    check("frag(benign) > frag(adv)", frag_b > frag_a,
          f"benign={frag_b:.3f}, adv={frag_a:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        p.save(path)

        model = AlwaysZero()
        x_b_tensor = torch.from_numpy(x_benign_np)
        x_adv_tensor = torch.from_numpy(x_adv_np)

        # mode="log"
        w = SentinelWrapper(model, str(path), mode="log", threshold=1.5)
        w.eval()
        with torch.no_grad():
            y = w(x_b_tensor)
        check("log: forward работает", y.shape == (40, 10))
        check("log: stats не пуст", w.stats()["n_total"] == 40)

        # mode="raise" на adversarial
        w2 = SentinelWrapper(model, str(path), mode="raise", threshold=1.5)
        w2.eval()
        try:
            with torch.no_grad():
                _ = w2(x_adv_tensor)
            check("raise: не заблокировал (странно)", False)
        except AdversarialDetectedError as e:
            check("raise: заблокировал", True)
            check("raise: есть indices", len(e.indices) > 0)

        # mode="zero" на adversarial
        w3 = SentinelWrapper(model, str(path), mode="zero", threshold=1.5)
        w3.eval()
        with torch.no_grad():
            y = w3(x_adv_tensor)
        n_zero = int((y.abs().sum(dim=1) == 0).sum())
        check("zero: обнулил хотя бы один", n_zero > 0,
              f"n_zero={n_zero}")


# ── main ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sentinel-Q — unit tests")
    print("=" * 60)

    test_core()
    test_qorb_descriptor()
    test_profile_build_save_load()
    test_profile_zscores()
    test_detector_modes()
    test_detector_batch()
    test_server_imports()
    test_torch_wrapper()

    print("\n" + "=" * 60)
    print(f"Итог: {len(PASSED)} OK, {len(FAILED)} FAIL")
    if FAILED:
        print("\nПровалы:")
        for name, detail in FAILED:
            print(f"  - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())