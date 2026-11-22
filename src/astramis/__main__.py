"""Запуск CPU-инференса без обучения и сетевых обращений."""

import argparse
import hashlib
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Каталог dataset")
    parser.add_argument("--sample", type=Path, required=True, help="Официальный sample_submission.csv")
    parser.add_argument("--models", type=Path, required=True, help="Файлы моделей и artifacts.json")
    parser.add_argument("--dino-source", type=Path, required=True, help="Распакованный репозиторий DINOv2")
    parser.add_argument("--output", type=Path, required=True, help="Путь к новому CSV")
    parser.add_argument("--reference", type=Path, help="Эталонный CSV для побайтовой сверки")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output должен указывать на новый файл")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["YOLO_OFFLINE"] = "true"

    from .pipeline import run

    results = run(args.data, args.sample, args.models, args.dino_source)
    for stage, blob in results.items():
        print(f"{stage} SHA-256 {hashlib.sha256(blob).hexdigest()}")
    result = results["wood_volume"]
    with args.output.open("xb") as file:
        file.write(result)
    if args.reference:
        reference = args.reference.read_bytes()
        print(f"reference SHA-256 {hashlib.sha256(reference).hexdigest()}")
        if result != reference:
            print("CSV отличаются", file=sys.stderr)
            return 1
        print("CSV совпадают побайтово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
