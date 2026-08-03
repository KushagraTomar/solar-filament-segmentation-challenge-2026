"""Run inference on the MAGFiLO test set and write a submission CSV.

Uses mmdet's own `init_detector`/`inference_detector` (rather than hand-rolling
the preprocessing pipeline) so predictions go through the exact same
normalization/resize/rescale logic the model was trained and validated with --
`inference_detector` strips `LoadAnnotations` from the config's test pipeline
automatically (there's no GT for the real test set) and rescales predicted
masks back to each image's original resolution (2048x2048) by default, which
is what the fixed-size RLE submission format expects.

Usage:
    python -m src.scripts.predict \
        --config configs/pointrend_r50_fpn_magfilo.py \
        --checkpoint /home/kushagra/Downloads/epoch_12.pth \
        --test-images-dir data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --out outputs/submission.csv \
        --score-thr 0.3
"""
import argparse
import functools
from pathlib import Path

import torch
from mmdet.apis import inference_detector, init_detector

from src.mmdet_plugins.patches import apply_pointrend_test_patch
from src.utils.rle import SubmissionWriter

apply_pointrend_test_patch()

# torch >=2.6 defaults torch.load to weights_only=True, which rejects mmengine
# checkpoints' non-tensor objects (HistoryBuffer, numpy internals from the
# optimizer/message_hub state, ...). Allowlisting classes one at a time is
# unbounded -- just disable the restriction for torch.load calls made from
# here on. Safe since we trust this checkpoint (produced by our own training
# run), not an arbitrary downloaded file.
torch.load = functools.partial(torch.load, weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-images-dir", required=True)
    parser.add_argument("--out", default="outputs/submission.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3,
                        help="Min instance score to keep. The model's own test_cfg.rcnn.score_thr "
                             "(0.05) is tuned for mAP-style eval with many low-score boxes pruned "
                             "later by NMS/top-k; for a Dice-scored submission a stricter cutoff "
                             "avoids flooding the CSV with low-confidence noise instances.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N test images (for a quick sanity check).")
    args = parser.parse_args()

    model = init_detector(args.config, args.checkpoint, device=args.device)

    image_paths = sorted(Path(args.test_images_dir).glob("*.jpeg"))
    if args.limit:
        image_paths = image_paths[: args.limit]

    writer = SubmissionWriter()
    total_instances = 0
    failed = []
    for i, img_path in enumerate(image_paths, start=1):
        file_stem = img_path.stem
        try:
            result = inference_detector(model, str(img_path))
            pred = result.pred_instances

            keep = pred.scores >= args.score_thr
            masks = pred.masks[keep].cpu().numpy()

            writer.add_image(file_stem, list(masks))
            total_instances += len(masks)

            print(f"[{i}/{len(image_paths)}] {img_path.name}: {len(masks)} instances kept "
                  f"(of {len(pred.scores)} raw detections)")
        except torch.OutOfMemoryError:
            # A single crowded image shouldn't take down the whole run --
            # log it and move on with zero rows for this image (better an
            # incomplete-but-flagged submission than losing everything).
            print(f"[{i}/{len(image_paths)}] {img_path.name}: OOM, skipping (0 instances written)")
            failed.append(file_stem)
        finally:
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    df = writer.write(args.out)
    print(f"\nWrote {len(df)} instance rows across {len(image_paths)} images to {args.out}")
    print(f"Images with zero detections: "
          f"{len(image_paths) - df['filament_id'].str.rsplit('_', n=1).str[0].nunique()}")
    if failed:
        print(f"WARNING: {len(failed)} image(s) skipped due to OOM (0 rows written for these): "
              f"{failed}")
        print("Re-run with --device cpu to cover these (slower, but no memory ceiling).")


if __name__ == "__main__":
    main()
