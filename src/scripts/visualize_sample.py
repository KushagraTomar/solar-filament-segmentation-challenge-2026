"""Sanity-check visualization: overlay filament masks + spines on one H-Alpha image.

Usage:
    python -m src.scripts.visualize_sample \
        --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
        --images-dir data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images \
        --image-id 040301-20140609195854Bh \
        --out outputs/sample_overlay.png
"""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.coco_utils import MagfiloAnnotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--image-id", default=None, help="Specific image_id; default: first available")
    parser.add_argument("--out", default="outputs/sample_overlay.png")
    args = parser.parse_args()

    anns = MagfiloAnnotations(args.json)
    print("dataset summary:", anns.summary())

    image_id = args.image_id or next(iter(anns.images_by_id))
    im_info = anns.images_by_id[image_id]
    img_path = f"{args.images_dir}/{im_info['file_name']}"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    instances = anns.masks_for(image_id)
    print(f"image_id={image_id} file_name={im_info['file_name']} "
          f"shape={img.shape} num_filaments={len(instances)}")

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    rng = np.random.default_rng(0)
    for mask, cat, spine in instances:
        color = rng.uniform(80, 255, size=3)
        overlay[mask.astype(bool)] = 0.55 * overlay[mask.astype(bool)] + 0.45 * color

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay.astype(np.uint8)[..., ::-1])
    for mask, cat, spine in instances:
        if spine is not None and len(spine) > 1:
            ax.plot(spine[:, 0], spine[:, 1], "-", linewidth=1, color="cyan")
    ax.set_title(f"{im_info['file_name']}  ({len(instances)} filaments)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
