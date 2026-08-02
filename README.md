# Solar Filament Segmentation (MAGFiLO / Kaggle 2026)

Instance segmentation of solar filaments from H-Alpha images. Planned architecture:
Cascade Mask R-CNN + HTC-style semantic auxiliary branch + PointRend mask head,
with a Dice + clDice loss supervised against the dataset's annotated filament
spines. See project discussion for full architecture rationale.

## Status

Data pipeline is built and validated against the real downloaded dataset (707
train images / 1154 annotator-batch entries / 8199 filament annotations, 180
test images). Model training (mmdetection: Cascade R-CNN + HTC + PointRend) is
the next phase — not yet built.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The `mmengine`/`mmcv`/`mmdet` lines in `requirements.txt` are commented out —
install them per the [mmdetection install guide](https://mmdetection.readthedocs.io/en/latest/get_started.html)
once training moves to the cloud GPU, since `mmcv` needs a build matched to
your CUDA/torch version.

Note: local GPU is a 4GB GTX 1650 — not enough for the full Cascade+HTC+PointRend
stack at 2048x2048. Local runs are for data-pipeline dev/debugging only; actual
training targets a cloud GPU.

## Project layout

```
data/
  raw/MAGFiLO_1.0_Kaggle_2026/   # unzipped Kaggle download (train/test images + train annotations JSON)
  splits/                        # generated train/val split files (see below)
src/
  data/
    coco_utils.py   # COCO JSON loading, polygon->mask, spine parsing/rasterization
    split.py        # train/val split by file_name (not image_id -- see below)
  losses/
    cldice.py        # Dice + clDice loss, spine-aware
  utils/
    rle.py           # mask <-> RLE, submission CSV writer
  scripts/
    make_split.py         # generates data/splits/*.json
    visualize_sample.py   # sanity-check overlay of masks+spines on an image
outputs/              # visualization / prediction outputs
```

## Key dataset gotchas this pipeline handles

1. **Multiple annotator batches per image.** The same underlying capture
   (`file_name`) can have multiple independent annotation sets
   (`image_id = "<batch>-<file_name>"`). `split.py` splits by `file_name`, not
   `image_id`, so near-duplicate frames never end up on both sides of
   train/val.
2. **Spine as ground-truth skeleton.** `annotation["spine"]` is a hand-annotated
   centerline. `coco_utils.rasterize_spine` turns it into a thin binary raster
   used directly as clDice's ground-truth skeleton (`gt_skeleton` arg), instead
   of approximating a skeleton from the mask via soft-erosion — more accurate,
   and free since it's already annotated.
3. **Instrument stratification.** File names encode a 2-letter GONG site code
   (e.g. `Bh`, `Lh`). The split is stratified by this code so both train and
   val see a representative mix of observatories (6 sites, roughly balanced:
   Lh/Mh/Bh/Ch ~200 each, Th/Uh ~170).
4. **Submission RLE.** Only `counts` is stored (size is fixed 2048x2048).
   `rle.py` round-trips exactly against real MAGFiLO masks (`mask_to_rle_counts`
   / `rle_counts_to_mask`).

## Usage

```bash
# Generate train/val split (stratified by instrument, split by file_name)
python -m src.scripts.make_split \
  --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --out data/splits/split_v1.json

# Visualize one image's masks + spines as a sanity check
python -m src.scripts.visualize_sample \
  --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --images-dir data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --image-id <image_id> \
  --out outputs/sample_overlay.png
```

## Next steps

- Wire up mmdetection config: register a custom COCO-format dataset class,
  Cascade R-CNN + HTC + PointRend config, and plug in `DiceClDiceLoss` (with
  rasterized spines) as the mask-head loss.
- Decide 2048x2048 input strategy (tile vs. resize vs. high-res-native backbone) --
  flagged earlier, not yet resolved.
- Training script + cloud GPU environment setup.
- Inference script: run trained model on `test/test_images`, convert predicted
  instance masks to RLE via `src/utils/rle.py`, write submission CSV.
