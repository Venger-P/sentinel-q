"""
text_qorb_v2.py — усиление сигнала для текстовых атак.

Отличия от v1:
  1. Тексты длиннее: 1000+ символов (объединение чанков).
  2. frag усреднён по 3 компрессорам (zlib, bz2, lzma).
  3. Добавлены word-level атаки (замена слов).
  4. Признаки: [frag_zlib, frag_bz2, frag_lzma, H, n_unique].

Ожидание: SNR вырастет с 0.67 до 1.5–2.0, AUC до 0.90+.

Запуск:
    python examples/text_qorb_v2.py
    python examples/text_qorb_v2.py --n 200 --rate 0.15
"""

import argparse
import importlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel_q.core import fragility, shannon_entropy, n_unique_bytes


# ── Загрузка корпуса ────────────────────────────────────────

def load_long_corpus(n_samples=200, min_len=1000):
    """
    Загружает корпус из docstrings Python, объединяя чанки
    в тексты длиной >= min_len символов.
    """
    modules = [
        'collections', 'itertools', 'functools', 're', 'json',
        'random', 'statistics', 'pathlib', 'typing', 'enum',
        'dataclasses', 'contextlib', 'string', 'textwrap',
        'argparse', 'subprocess', 'shutil', 'tempfile', 'os',
        'unittest', 'logging', 'socket', 'threading', 'asyncio',
    ]
    pieces = []

    for mod_name in modules:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue

        if isinstance(m.__doc__, str):
            pieces.append(m.__doc__)

        for name in dir(m):
            try:
                obj = getattr(m, name)
            except Exception:
                continue
            doc = getattr(obj, '__doc__', None)
            if isinstance(doc, str) and len(doc) > 300:
                pieces.append(doc)

    # Объединяем куски в длинные тексты
    random.seed(42)
    random.shuffle(pieces)

    corpus = []
    buf = ""
    for piece in pieces:
        buf += " " + piece
        if len(buf) >= min_len:
            corpus.append(buf.strip())
            buf = ""
        if len(corpus) >= n_samples:
            break

    return corpus


# ── Атаки ───────────────────────────────────────────────────

HOMOGLYPHS = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у',
    'A': 'А', 'E': 'Е', 'O': 'О', 'P': 'Р', 'C': 'С',
    'X': 'Х',
}

# Слова для word-level атаки (простой набор)
SYNONYMS = {
    "return": "give", "value": "amount", "object": "thing",
    "string": "text", "function": "method", "argument": "param",
    "the": "a", "and": "plus", "is": "equals", "was": "were",
}


def attack_swap(text, rate=0.15, rng=None):
    rng = rng or random
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < rate:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)


def attack_insert(text, rate=0.15, rng=None):
    rng = rng or random
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < rate:
            out.append(rng.choice(alphabet))
    return ''.join(out)


def attack_delete(text, rate=0.15, rng=None):
    rng = rng or random
    return ''.join(ch for ch in text if rng.random() > rate)


def attack_homoglyph(text, rate=0.5, rng=None):
    rng = rng or random
    return ''.join(
        HOMOGLYPHS.get(ch, ch) if rng.random() < rate else ch
        for ch in text
    )


def attack_word_replace(text, rate=0.10, rng=None):
    """Word-level: заменяет слова на синонимы."""
    rng = rng or random
    words = text.split()
    out = []
    for w in words:
        key = w.lower().strip('.,;:()[]{}"\'')
        if key in SYNONYMS and rng.random() < rate:
            # Сохраняем регистр первой буквы
            repl = SYNONYMS[key]
            if w and w[0].isupper():
                repl = repl.capitalize()
            out.append(repl)
        else:
            out.append(w)
    return ' '.join(out)


ATTACKS = {
    "swap":          attack_swap,
    "insert":        attack_insert,
    "delete":        attack_delete,
    "homoglyph":     attack_homoglyph,
    "word_replace":  attack_word_replace,
}


# ── Мульти-компрессорные признаки ───────────────────────────

def text_features_v2(text: str) -> dict:
    """
    Характеристики текста с frag по трём компрессорам.
    """
    data = text.encode('utf-8')
    frag_z = fragility(data, "zlib")
    frag_b = fragility(data, "bz2")
    frag_l = fragility(data, "lzma")
    return {
        "frag":     frag_z,
        "frag_z":   frag_z,
        "frag_b":   frag_b,
        "frag_l":   frag_l,
        "frag_avg": (frag_z + frag_b + frag_l) / 3.0,
        "H":        shannon_entropy(data),
        "n_unique": n_unique_bytes(data),
        "size":     len(data),
    }


# ── Эксперимент ─────────────────────────────────────────────

def run(n_samples=200, attack_rate=0.15):
    print("=" * 74)
    print("Sentinel-Q на тексте — версия 2 (усиленный сигнал)")
    print("=" * 74)

    # ── Корпус
    print(f"\n[1] Загрузка корпуса ({n_samples} длинных текстов) ...")
    corpus = load_long_corpus(n_samples, min_len=1000)
    print(f"    Загружено: {len(corpus)} текстов")
    sizes = [len(t) for t in corpus]
    print(f"    Длина: min={min(sizes)}, "
          f"median={int(np.median(sizes))}, max={max(sizes)}")

    # ── Benign
    print(f"\n[2] Характеристики benign ...")
    feats_b = [text_features_v2(t) for t in corpus]

    frag_avg_b = np.array([f["frag_avg"] for f in feats_b])
    H_b = np.array([f["H"] for f in feats_b])
    nu_b = np.array([f["n_unique"] for f in feats_b])

    print(f"    frag_avg = {frag_avg_b.mean():.4f} ± {frag_avg_b.std():.4f}")
    print(f"    H        = {H_b.mean():.4f} ± {H_b.std():.4f}")
    print(f"    n_uniq   = {nu_b.mean():.1f} ± {nu_b.std():.1f}")

    # ── Атаки
    print(f"\n[3] Атаки (rate={attack_rate}):")
    header = (f"    {'attack':<14} {'frag_z':>8} {'frag_b':>8} "
              f"{'frag_l':>8} {'frag_avg':>10} {'Δfrag':>9} {'ΔH':>8}")
    print(header)
    print("    " + "-" * (len(header) - 4))

    attacks_data = {}
    for name, fn in ATTACKS.items():
        rng = random.Random(42)
        attacked = [fn(t, rate=attack_rate, rng=rng) for t in corpus]
        feats = [text_features_v2(t) for t in attacked]

        fz = np.array([f["frag_z"] for f in feats])
        fb = np.array([f["frag_b"] for f in feats])
        fl = np.array([f["frag_l"] for f in feats])
        favg = np.array([f["frag_avg"] for f in feats])
        Ha = np.array([f["H"] for f in feats])

        attacks_data[name] = {
            "frag_z": fz, "frag_b": fb, "frag_l": fl,
            "frag_avg": favg, "H": Ha,
            "n_unique": np.array([f["n_unique"] for f in feats]),
        }

        print(f"    {name:<14} {fz.mean():>8.4f} {fb.mean():>8.4f} "
              f"{fl.mean():>8.4f} {favg.mean():>10.4f} "
              f"{favg.mean() - frag_avg_b.mean():>+9.4f} "
              f"{Ha.mean() - H_b.mean():>+8.4f}")

    # ── Детектор
    print(f"\n[4] Детектор (LR на [frag_z, frag_b, frag_l, H, n_unique]):")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        print("    sklearn не установлен — пропуск")
        return

    X_b = np.column_stack([
        [f["frag_z"] for f in feats_b],
        [f["frag_b"] for f in feats_b],
        [f["frag_l"] for f in feats_b],
        H_b, nu_b,
    ])

    header = (f"    {'attack':<14} {'AUC':>8} {'Δfrag_avg':>11} "
              f"{'SNR':>8} {'вывод':<10}")
    print(header)
    print("    " + "-" * (len(header) - 4))

    results = []
    for name, d in attacks_data.items():
        X_a = np.column_stack([
            d["frag_z"], d["frag_b"], d["frag_l"],
            d["H"], d["n_unique"],
        ])
        X = np.vstack([X_b, X_a])
        y = np.concatenate([np.zeros(len(X_b)), np.ones(len(X_a))])

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

        dfrag = d["frag_avg"].mean() - frag_avg_b.mean()
        std_frag = frag_avg_b.std()
        snr = abs(dfrag) / std_frag if std_frag > 0 else 0.0

        if auc > 0.95: v = "отлично"
        elif auc > 0.85: v = "хорошо"
        elif auc > 0.7: v = "средне"
        else: v = "слабо"

        print(f"    {name:<14} {auc:>8.4f} {dfrag:>+11.4f} "
              f"{snr:>8.3f} {v:<10}")
        results.append((name, auc, snr))

    # ── Сводка
    print(f"\n[5] Сводка:")
    avg_auc = np.mean([r[1] for r in results])
    avg_snr = np.mean([r[2] for r in results])
    print(f"    Средний AUC:  {avg_auc:.4f}")
    print(f"    Средний SNR:  {avg_snr:.3f}")
    print(f"\n    Сравнение с v1 (rate=0.20):")
    print(f"      v1:  AUC = 0.84,  SNR = 0.67")
    print(f"      v2:  AUC = {avg_auc:.3f}, SNR = {avg_snr:.3f}")
    if avg_auc > 0.90:
        print(f"    ✓ Усиление сработало: AUC > 0.90")
    elif avg_auc > 0.85:
        print(f"    ✓ Умеренное улучшение")
    else:
        print(f"    ~ Сигнал не усилился существенно")

    # ── Shuffle
    print(f"\n[6] Shuffle control:")
    rng = np.random.default_rng(42)
    shuffle_frag = []
    for t in corpus:
        data = t.encode('utf-8')
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        rng.shuffle(arr)
        shuffle_frag.append(fragility(arr.tobytes(), "zlib"))
    shuffle_frag = np.array(shuffle_frag)
    dfrag_sh = (frag_avg_b.mean() - shuffle_frag.mean())
    print(f"    Δfrag после shuffle: {dfrag_sh:+.4f}")
    if dfrag_sh > 0.01:
        print(f"    ✓ frag измеряет структуру (не только распределение)")

    print("\nГотово.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rate", type=float, default=0.15)
    args = ap.parse_args()

    run(n_samples=args.n, attack_rate=args.rate)


if __name__ == "__main__":
    main()