"""Kaggle Notebook training script for the MAGFiLO PointRend model.

Not run automatically -- paste these cells into a Kaggle Notebook (GPU
enabled). This is NOT a code-competition submission notebook (the
competition just wants a CSV upload), so internet access can stay on and
there's no kernel-time-limit-vs-inference constraint to worry about here --
this notebook is purely a free/cheap GPU training environment.

Before running:
  1. Upload outputs/kaggle_package/magfilo_code.zip as a new private Kaggle
     Dataset (Kaggle web UI: "New Dataset" -> upload the zip -- it auto-extracts).
     Attach it to this notebook as an input, alongside the competition's own
     MAGFiLO dataset.
  2. Check the actual mount paths under /kaggle/input/ (the exact folder name
     depends on the dataset slugs Kaggle assigns) and adjust CODE_DIR /
     DATA_ROOT below to match -- these are guesses based on typical Kaggle
     conventions, not verified against a real Kaggle environment.
"""

# ---- Cell 1: paths -- ADJUST these after checking /kaggle/input/ ----
CODE_DIR = "/kaggle/input/datasets/kushagratomar123/magfilo-2026"  # wherever the uploaded code dataset lands
DATA_ROOT = "/kaggle/input/notebooks/noob786/solar-filament-segmentation-challenge-2026"  # wherever the competition dataset lands
WORK_DIR = "/kaggle/working"

# ---- Cell 2: install deps ----
# !pip install -q pycocotools opencv-python-headless pandas scikit-learn matplotlib
# !pip install -q "setuptools<81" wheel  # PIN below 81, do not -U/upgrade: newer
#                                          # setuptools dropped pkg_resources by default,
#                                          # which mmcv/mmdet's setup.py still imports --
#                                          # `-U` grabs the latest and reintroduces the
#                                          # exact "egg_info did not run successfully" /
#                                          # "No module named 'pkg_resources'" error this
#                                          # is meant to avoid (see README's mmdetection
#                                          # install gotchas -- hit locally the same way).
# !pip install -q --no-build-isolation mmengine "mmcv==2.1.0" "mmdet==3.3.0"
# Kaggle GPU notebooks ship a preinstalled torch/CUDA combo -- do NOT reinstall
# torch/torchvision here the way the local README does (that reinstall was for
# a different problem: a plain `pip install torch` grabbing a CUDA build newer
# than the local machine's driver).
#
# BUT: `torch.cuda.is_available()` is NOT sufficient to verify things will
# work -- it was True and training still failed with `AcceleratorError: no
# kernel image is available for execution on the device`, because Kaggle's
# preinstalled torch build (2.10.0+cu128) doesn't include compiled kernels for
# the Tesla P100 (Pascal, sm_60) accelerator option -- only sm_70 and newer
# (check `torch.__config__.show()` / the build's NVCC gencode flags for the
# list). Recent PyTorch releases have dropped Pascal support; P100 is old
# enough to have fallen off.
# Fix: in the Kaggle notebook's settings panel, use accelerator "GPU T4 x2"
# instead of "GPU P100" -- T4 is Turing (sm_75), which IS supported. Restart
# the session after switching.

# ---- Cell 3: put the uploaded code on sys.path ----
import sys
sys.path.insert(0, CODE_DIR)

# ---- Cell 4: materialize the train/val split against the Kaggle-mounted data ----
# `!python -m ...` launches a NEW subprocess, which does NOT inherit Cell 3's
# in-process `sys.path.insert(0, CODE_DIR)` -- that only affects this kernel's
# own Python process. Subprocesses DO inherit environment variables though, so
# set PYTHONPATH on the command itself instead (fixes "No module named 'src'").
# !PYTHONPATH={CODE_DIR} python -m src.scripts.make_split \
#     --json {DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
#     --out {WORK_DIR}/split_v1.json
# !PYTHONPATH={CODE_DIR} python -m src.scripts.make_split_coco_jsons \
#     --json {DATA_ROOT}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
#     --split {WORK_DIR}/split_v1.json \
#     --out-dir {WORK_DIR}

# ---- Cell 5: point the config at the Kaggle paths, then train for real ----
# The config's ann_file paths (data/splits/{train,val}_coco.json, relative to
# data_root via '../../../../' traversal) assume the local repo layout -- on
# Kaggle, override data_root/ann_file directly instead of relying on that
# traversal, since the mount layout is different.
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile(f"{CODE_DIR}/configs/pointrend_r50_fpn_magfilo.py")
cfg.work_dir = WORK_DIR
cfg.train_dataloader.dataset.data_root = f"{DATA_ROOT}/train/"
cfg.train_dataloader.dataset.ann_file = f"{WORK_DIR}/train_coco.json"
cfg.train_dataloader.dataset.data_prefix = dict(img="train_images/")
# val disabled until the custom Dice/IoU eval script exists (see README) --
# same as the local training script.
cfg.val_cfg = None
cfg.val_dataloader = None
cfg.val_evaluator = None
cfg.test_cfg = None
cfg.test_dataloader = None
cfg.test_evaluator = None

# Real batch size -- Kaggle GPUs (P100/T4, 16GB) have much more headroom than
# the local 4GB card; raise this and re-check memory usage, don't just reuse
# the local confidence-check settings (batch_size=1, 512x512 crop).
cfg.train_dataloader.batch_size = 4
base_batch_size = cfg.get("auto_scale_lr", {}).get("base_batch_size", 16)
cfg.optim_wrapper.optimizer.lr *= cfg.train_dataloader.batch_size / base_batch_size

runner = Runner.from_cfg(cfg)
runner.train()

# ---- Cell 6: after training, checkpoints are under {WORK_DIR}/ --
# download the final .pth (Kaggle's "Output" tab) before the session ends.
