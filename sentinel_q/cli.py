"""
CLI для Sentinel-Q.

Команды:
    sentinel-q stats    <file>                      — байтовые характеристики
    sentinel-q profile  --data <npz> --out <json>   — построить профиль
    sentinel-q detect   --profile <json> --input <npz>  — детекция
    sentinel-q classify --train <npz> --test <npz>  — классификация атак
    sentinel-q serve    --profile <json>            — HTTP-сервер

Примеры:
    sentinel-q stats myfile.bin
    sentinel-q profile --data adv_cache/benign.npz --out profile_mnist.json
    sentinel-q detect --profile profile_mnist.json --input adv_cache/adversarial.npz
    sentinel-q detect --profile profile_mnist.json --input adv_cache/adversarial.npz --combine frag_only
    sentinel-q classify --train train.npz --test test.npz
    sentinel-q serve --profile profile_mnist.json --host 127.0.0.1 --port 8000
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ── Утилиты ─────────────────────────────────────────────────

def _npz_to_bytes(x):
    """
    Преобразует (N, C, H, W) float [0,1] в список bytes.

    Если C == 1, канал сжимается (получаем 2D представление).
    Иначе — сохраняются все каналы.
    """
    arr = (x * 255).astype(np.uint8)
    out = []
    for i in range(len(arr)):
        if arr.ndim == 4 and arr.shape[1] == 1:
            out.append(arr[i, 0].tobytes())
        else:
            out.append(arr[i].tobytes())
    return out


# ── Команды ─────────────────────────────────────────────────

def cmd_stats(args):
    """Байтовые характеристики файла."""
    from .core import byte_stats

    path = Path(args.file)
    if not path.exists():
        print(f"Файл не найден: {path}")
        sys.exit(1)

    data = path.read_bytes()
    st = byte_stats(data)

    print(f"Файл: {args.file}")
    print(f"  размер:    {st['size']} байт")
    print(f"  frag:      {st['frag']:.4f}")
    print(f"  frag_z:    {st['frag_z']:.4f}")
    print(f"  frag_b:    {st['frag_b']:.4f}")
    print(f"  H:         {st['H']:.4f} бит")
    print(f"  n_unique:  {st['n_unique']}")


def cmd_profile(args):
    """Построить эталонный профиль из .npz с x, y."""
    from .profile import ReferenceProfile

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Файл не найден: {data_path}")
        sys.exit(1)

    d = np.load(data_path)
    if "x" not in d or "y" not in d:
        print("В .npz должны быть ключи 'x' и 'y'")
        sys.exit(1)

    x, y = d["x"], d["y"]
    if args.limit:
        x, y = x[:args.limit], y[:args.limit]

    n_classes = len(set(y.tolist()))
    print(f"Строю профиль: {len(x)} примеров, {n_classes} классов")

    samples = _npz_to_bytes(x)
    profile = ReferenceProfile().build(samples, y.tolist(), verbose=True)
    profile.save(args.out)

    print(f"\nСохранено: {args.out}")
    print(profile.summary())


def cmd_detect(args):
    """Детекция adversarial-примеров по профилю."""
    from .profile import ReferenceProfile
    from .detector import SentinelDetector

    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"Профиль не найден: {profile_path}")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Входной файл не найден: {input_path}")
        sys.exit(1)

    profile = ReferenceProfile.load(profile_path)
    detector = SentinelDetector(
        profile,
        z_threshold=args.threshold,
        combine=args.combine,
    )

    d = np.load(input_path)
    if "x" not in d or "y" not in d:
        print("В .npz должны быть ключи 'x' и 'y'")
        sys.exit(1)

    x, y = d["x"], d["y"]
    if args.limit:
        x, y = x[:args.limit], y[:args.limit]

    samples = _npz_to_bytes(x)
    verdicts = detector.check_batch(samples, y.tolist())

    n_adv = sum(1 for v in verdicts if v.is_adversarial)
    mean_score = float(np.mean([v.score for v in verdicts]))

    print(f"Проверено: {len(verdicts)}")
    print(f"Подозрительных: {n_adv} ({n_adv/len(verdicts)*100:.1f}%)")
    print(f"Средний score: {mean_score:.3f}")
    print(f"combine = {args.combine}, threshold = {args.threshold}")

    if args.verbose:
        print("\nПервые 20 вердиктов:")
        for i, v in enumerate(verdicts[:20]):
            mark = "ADV" if v.is_adversarial else "ok "
            print(f"  [{mark}] #{i} score={v.score:+.3f}  {v.reason}")


def cmd_classify(args):
    """Классификация типа атаки по байтовым характеристикам."""
    from .classifier import AttackClassifier

    train_path = Path(args.train)
    test_path = Path(args.test)
    for p in (train_path, test_path):
        if not p.exists():
            print(f"Файл не найден: {p}")
            sys.exit(1)

    train = np.load(train_path)
    test = np.load(test_path)

    if "labels" not in train or "labels" not in test:
        print("В .npz должен быть ключ 'labels' "
              "(числовые метки типов атак)")
        sys.exit(1)

    samples_tr = _npz_to_bytes(train["x"])
    labels_tr = train["labels"].tolist()
    samples_te = _npz_to_bytes(test["x"])
    labels_te = test["labels"].tolist()

    print(f"Обучение на {len(samples_tr)} примеров, "
          f"{len(set(labels_tr))} классов ...")
    clf = AttackClassifier().fit(samples_tr, labels_tr)

    acc = clf.score(samples_te, labels_te)
    print(f"Accuracy (test): {acc:.4f}")

    if args.out:
        clf.save(args.out)
        print(f"Сохранено: {args.out}")

    if args.verbose:
        print("\nПримеры предсказаний (первые 10):")
        for i in range(min(10, len(samples_te))):
            pred = clf.predict(samples_te[i])
            proba = clf.predict_proba(samples_te[i])
            top_class = max(proba, key=proba.get)
            print(f"  #{i}: true={labels_te[i]}, pred={pred}, "
                  f"p={proba[top_class]:.3f}")


def cmd_serve(args):
    """Запустить HTTP-сервер."""
    try:
        import uvicorn
    except ImportError:
        print("Установите серверные зависимости: "
              "pip install sentinel-q[server]")
        sys.exit(1)

    from .server import create_app

    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"Профиль не найден: {profile_path}")
        sys.exit(1)

    app = create_app(
        profile_path=str(profile_path),
        default_combine=args.combine,
        default_threshold=args.threshold,
    )

    print(f"Запуск Sentinel-Q сервера")
    print(f"  profile:   {profile_path}")
    print(f"  host:      {args.host}")
    print(f"  port:      {args.port}")
    print(f"  combine:   {args.combine}")
    print(f"  threshold: {args.threshold}")
    print(f"  docs:      http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


# ── Основной парсер ─────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sentinel-q",
        description="Sentinel-Q — model-free adversarial detection",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # stats
    p = sub.add_parser("stats", help="Байтовые характеристики файла")
    p.add_argument("file", help="путь к файлу")
    p.set_defaults(func=cmd_stats)

    # profile
    p = sub.add_parser("profile",
                       help="Построить эталонный профиль из данных")
    p.add_argument("--data", required=True,
                   help=".npz с ключами x (N,C,H,W) и y (N,)")
    p.add_argument("--out", required=True,
                   help="путь для сохранения профиля (.json)")
    p.add_argument("--limit", type=int, default=None,
                   help="использовать только первые N примеров")
    p.set_defaults(func=cmd_profile)

    # detect
    p = sub.add_parser("detect",
                       help="Детектировать adversarial-примеры")
    p.add_argument("--profile", required=True,
                   help="путь к профилю (.json)")
    p.add_argument("--input", required=True,
                   help=".npz с x (N,C,H,W) и y (N,)")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="порог score (по умолчанию 1.5)")
    p.add_argument("--combine", default="weighted",
                   choices=["weighted", "frag_only", "mean", "max"],
                   help="способ комбинирования z-оценок")
    p.add_argument("--limit", type=int, default=None,
                   help="использовать только первые N примеров")
    p.add_argument("--verbose", action="store_true",
                   help="показать первые 20 вердиктов")
    p.set_defaults(func=cmd_detect)

    # classify
    p = sub.add_parser("classify",
                       help="Классифицировать тип атаки")
    p.add_argument("--train", required=True,
                   help=".npz с x и labels (числовые метки)")
    p.add_argument("--test", required=True,
                   help=".npz с x и labels")
    p.add_argument("--out", default=None,
                   help="сохранить обученный классификатор (.pkl)")
    p.add_argument("--verbose", action="store_true",
                   help="показать примеры предсказаний")
    p.set_defaults(func=cmd_classify)

    # serve
    p = sub.add_parser("serve",
                       help="Запустить HTTP-сервер")
    p.add_argument("--profile", required=True,
                   help="путь к профилю (.json)")
    p.add_argument("--host", default="127.0.0.1",
                   help="адрес (по умолчанию 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000,
                   help="порт (по умолчанию 8000)")
    p.add_argument("--combine", default="weighted",
                   choices=["weighted", "frag_only", "mean", "max"],
                   help="default combine")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="default threshold")
    p.add_argument("--log-level", default="info",
                   choices=["critical", "error", "warning",
                            "info", "debug", "trace"],
                   help="уровень логов uvicorn")
    p.set_defaults(func=cmd_serve)

    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)
    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        sys.exit(1)


def serve_entry():
    """
    Точка входа для скрипта sentinel-q-serve.

    Эквивалент `sentinel-q serve ...`, позволяет запускать сервер
    без указания подкоманды:
        sentinel-q-serve --profile profile.json --port 8000
    """
    sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()