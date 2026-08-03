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
  3.3.0 requires `mmcv<2.2.0`). `--no-build-isolation` only helps if the
  *ambient* environment's own setuptools still has `pkg_resources` --
  explicitly run `pip install "setuptools<81" wheel` first (pin, don't `-U`/
  upgrade: upgrading grabs the latest setuptools, which is the exact thing
  that drops `pkg_resources` and reintroduces the "egg_info did not run
  successfully" error). This worked without the explicit pin locally only
  because the venv's ambient setuptools happened to already be old enough --
  don't rely on that in a fresh environment (bit us this way on a Kaggle
  notebook).
- **Check `torch.cuda.is_available()` right after installing torch.** A
  plain `pip install torch` can silently grab a `cuXXX` build newer than
  what your NVIDIA driver supports (this happened locally: driver
  550.163.01 only supports up to CUDA 12.4, but plain pip install grabbed a
  cu130 build) -- `cuda.is_available()` returns `False` with no error in
  that case. Install explicitly from the matching wheel index, e.g.
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`.
- **`torch.cuda.is_available() == True` is NOT sufficient** -- it can still be
  `True` while the installed torch build has no compiled kernels for your
  specific GPU's compute capability, failing later with `AcceleratorError: no
  kernel image is available for execution on the device` (hit this on a
  Kaggle notebook: the pre-installed `torch==2.10.0+cu128` only had kernels
  for `sm_70` and newer, but Kaggle's "GPU P100" accelerator option is Pascal,
  `sm_60` -- recent PyTorch releases have dropped Pascal support). Check the
  actual GPU name against the torch build's supported architectures (its
  NVCC gencode flags, visible in `torch.__config__.show()`) before training,
  not just whether CUDA initializes. On Kaggle specifically: prefer the
  "GPU T4 x2" accelerator (`sm_75`) over "GPU P100" for this reason.
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

```text
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
- **The config's `lr=0.02` assumes batch_size=16** (`auto_scale_lr.base_batch_size`
  in the config). Running at a smaller batch size with the raw LR -- and no
  warmup -- diverges to `NaN` within ~10 iterations on freshly-initialized
  detection heads. `src/scripts/train.py` now applies the linear LR scaling
  rule automatically and keeps the (iteration-based, `by_epoch=False`)
  warmup scheduler even when `--max-iters` overrides the epoch-based decay
  schedule -- verified fixed: loss now drops cleanly over 150 real iterations
  instead of NaN-ing out.
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

# Longer local confidence check -- confirms the loss actually decreases (not just
# "no crash"). Applies the linear LR-scaling rule for whatever batch size you pass,
# and keeps the iteration-based warmup even when overriding the epoch schedule --
# skipping either of those diverges to NaN within ~10 iterations (this bit us once).
python -m src.scripts.train --config configs/pointrend_r50_fpn_magfilo.py \
  --work-dir outputs/train_confidence_check --max-iters 150 --crop 512 --batch-size 1 --log-interval 10
```

### Kaggle Notebook (real training)

`notebooks/kaggle_train.py` has the cell-by-cell setup for running real
training on Kaggle's free GPU notebooks (this competition is CSV-upload
only, not a code competition, so there's no internet-off / kernel-submission
constraint here -- the notebook is just a GPU compute environment). Steps:

1. Zip the code: `zip -r outputs/kaggle_package/magfilo_code.zip src configs requirements.txt README.md -x "*__pycache__*"`
   (already done once, in `outputs/kaggle_package/magfilo_code.zip`).
2. Upload that zip as a new private Kaggle Dataset via the Kaggle web UI --
   not automated here, since it needs your Kaggle account credentials.
3. Create a new Notebook, attach your uploaded code dataset + this
   competition's dataset as inputs, enable GPU.
4. Check the actual mount paths under `/kaggle/input/` (exact folder names
   depend on the dataset slugs Kaggle assigns) and adjust `CODE_DIR` /
   `DATA_ROOT` at the top of `notebooks/kaggle_train.py` to match -- these
   are reasonable guesses, not verified against a real Kaggle environment.
5. Paste the rest of the cells in and run.
6. Before the session ends, download the checkpoint from the Kaggle
   "Output" tab -- Kaggle notebook sessions aren't persistent storage.

## Inference / submission

`src/scripts/predict.py` runs a trained checkpoint over the test set and
writes the submission CSV, using mmdet's own `init_detector`/
`inference_detector` (same preprocessing/rescale path as training/val,
rather than a hand-rolled reimplementation):

```bash
python -m src.scripts.predict \
  --config configs/pointrend_r50_fpn_magfilo.py \
  --checkpoint /path/to/epoch_12.pth \
  --test-images-dir data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --out outputs/submission.csv \
  --score-thr 0.3
```

Verified end-to-end against a real 12-epoch Kaggle-trained checkpoint. Two
more real bugs found and fixed along the way, both in
`src/mmdet_plugins/patches.py` and this script:

- **A second, deeper consequence of `CoarseMaskHead`'s `class_agnostic` bug**
  (see the mmdet gotchas above) -- it also breaks mmdet's own
  `PointRendRoIHead._mask_point_forward_test` at inference time (not just the
  training loss): the point head's single-channel refinement can't be
  `scatter_`ed into the coarse head's still-multi-channel prediction as mmdet
  wrote it. `apply_pointrend_test_patch()` fixes this by keeping the coarse
  features at their real (4-channel) width where `MaskPointHead`'s trained fc
  layers need it, but writing the refined value into only the predicted-class
  channel per instance, then reducing to that one channel right before
  `_predict_by_feat_single` (which otherwise assumes the tensor's already
  reduced when `class_agnostic=True`). See the file's docstring for the full
  chain of reasoning -- this took three iterations to get right.
- **torch >=2.6 checkpoint loading**: defaults `torch.load` to
  `weights_only=True`, which rejects mmengine's own non-tensor checkpoint
  contents (`HistoryBuffer`, numpy internals in the optimizer state).
  Allowlisting classes one at a time is unbounded; `predict.py` just disables
  the restriction outright, safe here since it's our own trusted checkpoint.
- **Local 4GB GPU OOMs on crowded images** (many instances pasted into a
  2048x2048 canvas at once) -- `predict.py` clears the CUDA cache between
  images (fixed it in practice) and catches OOM per-image as a fallback
  (skips that image with a warning rather than losing the whole run); pass
  `--device cpu` for a slower but memory-ceiling-free run if needed.

## Next steps

- Run inference over the full 180-image test set (only sanity-checked on a
  handful of images so far) and do a final review of the submission CSV
  before uploading.
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
- Consider the U-Net-as-refiner design discussed (per-instance crop
  refinement stage on top of PointRend's output, giving clDice/spine
  supervision a tractable home) -- not yet built.
