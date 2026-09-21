"""
text_qorb.py — Sentinel-Q на тексте.

Проверяет, работает ли метрика frag на текстовых данных.

Корпус: docstrings стандартной библиотеки Python.
Атаки: character-level (swap, insert, delete, homoglyph).
Метрики: frag, H, n_unique.
Детектор: Logistic Regression + cross-validation.

Запуск:
    python examples/text_qorb.py
    python examples/text_qorb.py --n 500 --rate 0.05
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

def split_into_chunks(text, n=3):
    """Разбивает текст на n примерно равных кусков по словам."""
    words = text.split()
    if len(words) < 30:
        return []
    chunk_size = max(30, len(words) // n)
    return [' '.join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)][:n]


def load_corpus(n_samples=300):
    """
    Загружает корпус из docstrings стандартной библиотеки Python.

    Это реальный английский текст со структурой — не случайные
    символы. Он сжимается лучше, чем случайный набор букв.
    """
    modules = [
        'collections', 'itertools', 'functools', 're', 'json',
        'random', 'statistics', 'pathlib', 'typing', 'enum',
        'dataclasses', 'contextlib', 'string', 'textwrap',
        'argparse', 'subprocess', 'shutil', 'tempfile', 'os',
    ]
    texts = []

    for mod_name in modules:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue

        # Документация модуля
        if isinstance(m.__doc__, str):
            texts.extend(split_into_chunks(m.__doc__, 3))

        # Документация объектов внутри модуля
        for name in dir(m):
            try:
                obj = getattr(m, name)
            except Exception:
                continue
            doc = getattr(obj, '__doc__', None)
            # Сначала проверяем, что это строка, потом длину
            if isinstance(doc, str) and len(doc) > 200:
                texts.extend(split_into_chunks(doc, 2))

    random.seed(42)
    random.shuffle(texts)
    return texts[:n_samples]


# ── Атаки ───────────────────────────────────────────────────

HOMOGLYPHS = {
    'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с',
    'x': 'х', 'y': 'у',
    'A': 'А', 'E': 'Е', 'O': 'О', 'P': 'Р', 'C': 'С',
    'X': 'Х',
}


def attack_swap(text, rate=0.05, rng=None):
    """Меняет местами соседние символы."""
    rng = rng or random
    chars = list(text)
    for i in range(len(chars) - 1):
        if rng.random() < rate:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return ''.join(chars)


def attack_insert(text, rate=0.05, rng=None):
    """Вставляет случайные буквы."""
    rng = rng or random
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    out = []
    for ch in text:
        out.append(ch)
        if rng.random() < rate:
            out.append(rng.choice(alphabet))
    return ''.join(out)


def attack_delete(text, rate=0.05, rng=None):
    """Удаляет случайные символы."""
    rng = rng or random
    return ''.join(ch for ch in text if rng.random() > rate)


def attack_homoglyph(text, rate=0.3, rng=None):
    """
    Заменяет латинские символы на визуально идентичные кириллические.

    Пример: 'a' (U+0061, 1 байт) → 'а' (U+0430, 2 байта в UTF-8).
    Текст выглядит идентично, но на уровне байтов это совершенно
    другое содержимое.
    """
    rng = rng or random
    return ''.join(
        HOMOGLYPHS.get(ch, ch) if rng.random() < rate else ch
        for ch in text
    )


ATTACKS = {
    "swap":       attack_swap,
    "insert":     attack_insert,
    "delete":     attack_delete,
    "homoglyph":  attack_homoglyph,
}


# ── Метрики ─────────────────────────────────────────────────

def text_features(text: str) -> dict:
    """Считает байтовые характеристики текста."""
    data = text.encode('utf-8')
    return {
        "frag":     fragility(data, "zlib"),
        "H":        shannon_entropy(data),
        "n_unique": n_unique_bytes(data),
        "size":     len(data),
    }


# ── Основной эксперимент ────────────────────────────────────

def run_experiment(n_samples=300, attack_rate=0.05):
    print("=" * 72)
    print("Sentinel-Q на тексте — character-level атаки")
    print("=" * 72)

    # ── 1. Корпус
    print(f"\n[1] Загрузка корпуса ({n_samples} текстов) ...")
    corpus = load_corpus(n_samples)
    print(f"    Загружено: {len(corpus)} текстов")
    if not corpus:
        print("    ОШИБКА: корпус пуст")
        return

    sizes = [len(t) for t in corpus]
    print(f"    Длина: min={min(sizes)}, "
          f"median={int(np.median(sizes))}, max={max(sizes)}")

    # ── 2. Benign
    print(f"\n[2] Характеристики benign ...")
    benign_feats = [text_features(t) for t in corpus]
    frag_b = np.array([f["frag"] for f in benign_feats])
    H_b = np.array([f["H"] for f in benign_feats])
    nu_b = np.array([f["n_unique"] for f in benign_feats])

    print(f"    frag   = {frag_b.mean():.4f} ± {frag_b.std():.4f}")
    print(f"    H      = {H_b.mean():.4f} ± {H_b.std():.4f}")
    print(f"    n_uniq = {nu_b.mean():.1f} ± {nu_b.std():.1f}")

    # ── 3. Атаки
    print(f"\n[3] Атаки (rate={attack_rate}):")
    header = f"    {'attack':<12} {'frag':>8} {'Δfrag':>9} " \
             f"{'H':>8} {'ΔH':>9} {'n_uniq':>8}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    attacked_feats = {}
    for name, attack_fn in ATTACKS.items():
        rng = random.Random(42)
        attacked = [attack_fn(t, rate=attack_rate, rng=rng)
                    for t in corpus]
        feats = [text_features(t) for t in attacked]
        frag_a = np.array([f["frag"] for f in feats])
        H_a = np.array([f["H"] for f in feats])
        nu_a = np.array([f["n_unique"] for f in feats])

        attacked_feats[name] = {
            "frag": frag_a, "H": H_a, "n_unique": nu_a,
        }

        print(f"    {name:<12} {frag_a.mean():>8.4f} "
              f"{frag_a.mean() - frag_b.mean():>+9.4f} "
              f"{H_a.mean():>8.4f} "
              f"{H_a.mean() - H_b.mean():>+9.4f} "
              f"{nu_a.mean():>8.1f}")

    # ── 4. Детектор
    print(f"\n[4] Детектор (LR на [frag, H, n_unique], 5-fold CV):")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        X_b = np.column_stack([frag_b, H_b, nu_b])

        header = f"    {'attack':<12} {'AUC':>8} {'Δfrag':>9} {'вывод'}"
        print(header)
        print("    " + "-" * (len(header) - 4))

        for name in ATTACKS:
            af = attacked_feats[name]
            X_a = np.column_stack([af["frag"], af["H"], af["n_unique"]])
            X = np.vstack([X_b, X_a])
            y = np.concatenate([np.zeros(len(X_b)), np.ones(len(X_a))])

            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000,
                                   class_weight="balanced"),
            )
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            auc = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="roc_auc").mean()

            dfrag = af["frag"].mean() - frag_b.mean()
            if auc > 0.9:
                verdict = "отлично"
            elif auc > 0.75:
                verdict = "хорошо"
            elif auc > 0.6:
                verdict = "средне"
            else:
                verdict = "слабо"

            print(f"    {name:<12} {auc:>8.4f} {dfrag:>+9.4f} {verdict}")

    except ImportError:
        print("    sklearn не установлен — пропуск")

    # ── 5. Shuffle control
    print(f"\n[5] Shuffle control — главный тест:")
    print(f"    Если frag реагирует на перестановку байтов,")
    print(f"    значит он измеряет структуру, а не только распределение.")
    print()

    rng = np.random.default_rng(42)
    shuffle_frag = []
    shuffle_H = []

    for t in corpus:
        data = t.encode('utf-8')
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        rng.shuffle(arr)
        shuffled = arr.tobytes()
        shuffle_frag.append(fragility(shuffled, "zlib"))
        shuffle_H.append(shannon_entropy(shuffled))

    shuffle_frag = np.array(shuffle_frag)
    shuffle_H = np.array(shuffle_H)

    dH_shuffle = np.abs(shuffle_H - H_b).mean()
    dfrag_shuffle = (frag_b - shuffle_frag).mean()

    print(f"    ΔH после shuffle:     {dH_shuffle:.2e}  "
          f"(математически должно быть ~0)")
    print(f"    Δfrag после shuffle:  {dfrag_shuffle:+.4f}")

    if dfrag_shuffle > 0.01:
        print(f"    ✓ frag реагирует на перестановку — измеряет структуру")
        print(f"      Это значит: метрика работает на тексте так же, "
              f"как на изображениях.")
    elif dfrag_shuffle > 0.001:
        print(f"    ~ слабая реакция на перестановку")
        print(f"      Метрика работает, но эффект на тексте слабее.")
    else:
        print(f"    ✗ frag не реагирует на перестановку")
        print(f"      На тексте метрика измеряет только распределение байтов.")

    # ── 6. Сравнение с изображениями
    print(f"\n[6] Сравнение с MNIST:")
    print(f"    {'метрика':<20} {'MNIST':>12} {'текст':>12}")
    print("    " + "-" * 46)
    print(f"    {'Δfrag (shuffle)':<20} {'+0.121':>12} "
          f"{dfrag_shuffle:>+12.4f}")
    print(f"    {'H (benign)':<20} {'1.53':>12} {H_b.mean():>12.4f}")
    print(f"    {'frag (benign)':<20} {'0.340':>12} "
          f"{frag_b.mean():>12.4f}")

    print("\nГотово.")


# ── main ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300,
                    help="число текстов в корпусе")
    ap.add_argument("--rate", type=float, default=0.05,
                    help="интенсивность атаки (0.0-1.0)")
    args = ap.parse_args()

    run_experiment(n_samples=args.n, attack_rate=args.rate)


if __name__ == "__main__":
    main()