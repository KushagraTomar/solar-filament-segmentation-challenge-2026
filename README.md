# Solar Filament Segmentation (MAGFiLO / Kaggle 2026)

Instance segmentation of solar filaments from H-Alpha images.

Architecture: **PointRend** (Mask R-CNN + point-based mask boundary
refinement), chosen over a Cascade R-CNN + HTC combination because mmdet
implements PointRend as its own RoIHead (`PointRendRoIHead`, single-stage),
not as a mask-head type pluggable into `HybridTaskCascadeRoIHead`'s cascade
-- combining all three would require a custom RoIHead, which was descoped in
favor of PointRend's fine-boundary/barb recovery, the more direct fit for
this competition's challenges. See `configs/pointrend_r50_fpn_magfilo.py` for
the full rationale and open items.

## Status

Data pipeline is built and validated against the real downloaded dataset (707
train images / 1154 annotator-batch entries / 8199 filament annotations, 180
test images). The mmdetection PointRend config builds and trains real
iterations end-to-end on real data (verified locally, see
`src/scripts/smoke_test_config.py`). Not yet done: real training run, the
2048x2048 input-resolution decision, spine-aware clDice integration into the
per-RoI mask/point heads, inference + submission script.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### mmdetection install gotchas (already applied in this repo's `.venv`, documented here in case of a fresh install / cloud environment setup)

- `pip install mmcv`/`mmdet` needs `--no-build-isolation` (their setup.py
  imports `pkg_resources`, which recent `setuptools` no longer bundles by
  default in an isolated build env) and `mmcv==2.1.0` specifically (mmdet
  3.3.0 requires `mmcv<2.2.0`).
- **Check `torch.cuda.is_available()` right after installing torch.** A
  plain `pip install torch` can silently grab a `cuXXX` build newer than
  what your NVIDIA driver supports (this happened locally: driver
  550.163.01 only supports up to CUDA 12.4, but plain pip install grabbed a
  cu130 build) -- `cuda.is_available()` returns `False` with no error in
  that case. Install explicitly from the matching wheel index, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
- If you change the torch build after mmcv is already installed, **force a
  clean mmcv rebuild** (`pip uninstall mmcv && pip cache remove mmcv && pip
  install --no-cache-dir --no-build-isolation mmcv==2.1.0`) -- mmcv's
  compiled ops are ABI-linked to the exact torch build; pip will otherwise
  silently reuse the stale cached wheel and you'll get an `undefined symbol`
  ImportError from `mmcv._ext`.

Local GPU is a 4GB GTX 1650 -- real training targets a cloud GPU, but the
PointRend config (unlike the originally-discussed Cascade+HTC+PointRend
stack) is light enough to smoke-test locally at a small crop size (see
Usage below); GPU memory usage in that test was ~880MB.

## Project layout

```
data/
  raw/MAGFiLO_1.0_Kaggle_2026/   # unzipped Kaggle download (train/test images + train annotations JSON)
  splits/                        # generated train/val split + per-split COCO JSONs
configs/
  pointrend_r50_fpn_magfilo.py   # mmdet PointRend config (model, losses, dataloaders, schedule)
src/
  data/
    coco_utils.py   # COCO JSON loading, polygon->mask, spine parsing/rasterization
    split.py        # train/val split by file_name (not image_id -- see below)
  losses/
    cldice.py        # Dice + clDice loss, spine-aware (not yet wired into the mmdet config, see below)
  utils/
    rle.py           # mask <-> RLE, submission CSV writer
  mmdet_plugins/      # custom mmdet registry entries, imported via configs' `custom_imports`
    datasets/magfilo.py    # MagfiloDataset: 4 categories, preserves `spine` field through mmdet's loader
    transforms/semantic_gt.py  # BuildSemanticGT: unused by the current PointRend config (was for
                                # the descoped HTC semantic branch); kept for a possible future revisit
    losses/dice_loss.py    # MagfiloDiceLoss: fixes two real mmdet gotchas, see file docstring
  scripts/
    make_split.py               # generates data/splits/*.json
    make_split_coco_jsons.py    # materializes per-split standalone COCO JSON files for mmdet
    visualize_sample.py         # sanity-check overlay of masks+spines on an image
    smoke_test_config.py        # builds model+dataloader from an mmdet config, runs real training
                                 # iterations on a small crop -- use this to validate config changes
outputs/              # visualization / prediction / smoke-test outputs
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
   and free since it's already annotated. (Currently only used by the
   standalone `src/losses/cldice.py` module, not yet wired into the mmdet
   per-RoI mask heads -- see Next steps.)
3. **Instrument stratification.** File names encode a 2-letter GONG site code
   (e.g. `Bh`, `Lh`). The split is stratified by this code so both train and
   val see a representative mix of observatories (6 sites, roughly balanced:
   Lh/Mh/Bh/Ch ~200 each, Th/Uh ~170).
4. **Submission RLE.** Only `counts` is stored (size is fixed 2048x2048).
   `rle.py` round-trips exactly against real MAGFiLO masks (`mask_to_rle_counts`
   / `rle_counts_to_mask`).

## mmdet integration gotchas found while wiring the config

These are documented in the relevant files' docstrings too, but worth
flagging here since they're non-obvious upstream mmdet behaviors, not
MAGFiLO-specific:

- **String image/annotation IDs work fine.** MAGFiLO's COCO JSON uses string
  ids (e.g. `"040301-20140609195854Bh"`), not the usual ints. Verified both
  plain `pycocotools.COCO()` and mmdet's `CocoDataset` handle this correctly
  (dict-based indexing, agnostic to key type) -- but mmdet's built-in
  `CocoMetric` (pycocotools `COCOeval`) is not verified safe for this, so
  `val_evaluator`/`test_evaluator` are disabled in the config; validation
  will use a custom Dice/IoU script instead (not yet written).
- **`CoarseMaskHead` (used by PointRend) has a `class_agnostic` bug.** Its
  own `fc_logits` layer hardcodes `num_classes` output channels regardless
  of `class_agnostic=True` (unlike the base `FCNMaskHead`, which respects
  it correctly). `MaskPointHead` does not have this bug. See
  `src/mmdet_plugins/losses/dice_loss.py` docstring.
- **mmdet's built-in `DiceLoss` silently produces a zero loss under
  `class_agnostic=True`.** The mask/point heads always call
  `loss_mask(pred, target, label)`, where `label` is per-instance class index
  (all zeros under class_agnostic). `CrossEntropyLoss(use_mask=True)`
  interprets `label` as a channel selector; mmdet's `DiceLoss` instead
  treats that same positional argument as a per-sample *weight* and
  multiplies the loss by it -- silently zeroing the entire mask loss with no
  error. `MagfiloDiceLoss` (`src/mmdet_plugins/losses/dice_loss.py`) fixes
  both issues: it treats `label` as a channel selector, and is robust to
  `CoarseMaskHead` staying multi-channel despite `class_agnostic=True`.

## Usage

```bash
# Generate train/val split (stratified by instrument, split by file_name)
python -m src.scripts.make_split \
  --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --out data/splits/split_v1.json

# Materialize standalone per-split COCO JSONs (mmdet wants one ready-made file per split)
python -m src.scripts.make_split_coco_jsons \
  --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --split data/splits/split_v1.json \
  --out-dir data/splits

# Visualize one image's masks + spines as a sanity check
python -m src.scripts.visualize_sample \
  --json data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --images-dir data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --image-id <image_id> \
  --out outputs/sample_overlay.png

# Smoke-test the mmdet config end-to-end on real data (small crop, few iterations,
# safe to run on the local 4GB GPU; use after any config/plugin change)
python -m src.scripts.smoke_test_config --config configs/pointrend_r50_fpn_magfilo.py
```

## Next steps

- Decide the 2048x2048 input strategy (tile vs. resize vs. high-res-native
  backbone) -- flagged during architecture design, still open; the config
  currently uses a 1024x1024 resize placeholder.
- Wire the spine-aware clDice loss (`src/losses/cldice.py`) into
  `CoarseMaskHead`/`MaskPointHead`: needs the same per-RoI crop/resize
  applied to the rasterized spine that mmdet already applies to GT masks
  when building per-RoI targets.
- Write a custom Dice/IoU validation script matching the competition rubric
  (mean Dice, Dice/IoU distributions, one-to-many/many-to-one matching) --
  mmdet's built-in `CocoMetric` isn't a fit (see gotchas above).
- Real training run on a cloud GPU (local 4GB card is smoke-test only).
- Inference script: run trained model on `test/test_images`, convert
  predicted instance masks to RLE via `src/utils/rle.py`, write submission CSV.
