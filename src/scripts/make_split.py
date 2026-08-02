"""Generate the train/val split file from the MAGFiLO train annotations.

Usage:
    python -m src.scripts.make_split \
        --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --out data/splits/split_v1.json
"""
import argparse
from collections import Counter

from src.data.coco_utils import instrument_code
from src.data.split import write_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to train annotations JSON")
    parser.add_argument("--out", required=True, help="Output split JSON path")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split = write_split(args.json, args.out, val_fraction=args.val_fraction, seed=args.seed)

    train_instr = Counter(instrument_code(fn) for fn in split["train_file_names"])
    val_instr = Counter(instrument_code(fn) for fn in split["val_file_names"])

    print(f"train: {len(split['train_file_names'])} file_names, "
          f"{len(split['train_image_ids'])} image_ids")
    print(f"val:   {len(split['val_file_names'])} file_names, "
          f"{len(split['val_image_ids'])} image_ids")
    print(f"train instrument mix: {dict(sorted(train_instr.items()))}")
    print(f"val instrument mix:   {dict(sorted(val_instr.items()))}")
    print(f"wrote split to {args.out}")


if __name__ == "__main__":
    main()
