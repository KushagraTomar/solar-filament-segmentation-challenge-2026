"""Materialize per-split standalone COCO-format JSON files from the full
MAGFiLO train annotations + a split file (see split.py / make_split.py).

mmdet's CocoDataset expects one ready-made COCO JSON per split; this avoids
needing a custom filter_cfg/sampler and keeps the mmdet-facing dataset dead simple.

Usage:
    python -m src.scripts.make_split_coco_jsons \
        --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --split data/splits/split_v1.json \
        --out-dir data/splits
"""
import argparse
import json


def subset_coco(full, image_ids):
    image_id_set = set(image_ids)
    images = [im for im in full["images"] if im["id"] in image_id_set]
    annotations = [ann for ann in full["annotations"] if ann["image_id"] in image_id_set]
    return {
        "info": full["info"],
        "licenses": full["licenses"],
        "categories": full["categories"],
        "images": images,
        "annotations": annotations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to full train annotations JSON")
    parser.add_argument("--split", required=True, help="Path to split JSON (from make_split.py)")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.json) as f:
        full = json.load(f)
    with open(args.split) as f:
        split = json.load(f)

    for name in ("train", "val"):
        subset = subset_coco(full, split[f"{name}_image_ids"])
        out_path = f"{args.out_dir}/{name}_coco.json"
        with open(out_path, "w") as f:
            json.dump(subset, f)
        print(f"{name}: {len(subset['images'])} images, {len(subset['annotations'])} annotations -> {out_path}")


if __name__ == "__main__":
    main()
