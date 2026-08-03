"""Train the PointRend MAGFiLO model from a config.

Unlike smoke_test_config.py (which aggressively shrinks the config for a fast
correctness check), this runs the config close to as-authored. On the local
4GB GTX 1650 this is still only suitable for a small-scale confidence check
(override crop size / iteration count via flags below) -- real training
targets a cloud GPU, see README.md.

Usage:
    python -m src.scripts.train --config configs/pointrend_r50_fpn_magfilo.py \
        --work-dir outputs/train_v1 --max-iters 200 --crop 512 --batch-size 1
"""
import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", default="outputs/train")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="If set, overrides the config's epoch-based schedule with a "
                             "short IterBasedTrainLoop instead -- for local confidence checks, "
                             "not real training runs.")
    parser.add_argument("--crop", type=int, default=None,
                        help="If set, overrides the train pipeline's Resize scale (square).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size

    # Linear LR scaling rule: the config's base_lr assumes batch_size=16 (see
    # auto_scale_lr.base_batch_size). Any other batch size needs a scaled-down
    # LR, or training diverges to NaN within a handful of iterations -- this
    # bit us on the first confidence-check run at batch_size=1 with the raw
    # base_lr, since the warmup below got dropped entirely at the same time.
    base_batch_size = cfg.get("auto_scale_lr", {}).get("base_batch_size", 16)
    actual_batch_size = cfg.train_dataloader.batch_size
    scale = actual_batch_size / base_batch_size
    cfg.optim_wrapper.optimizer.lr *= scale

    if args.crop is not None:
        for step in cfg.train_dataloader.dataset.pipeline:
            if step["type"] == "Resize":
                step["scale"] = (args.crop, args.crop)
    if args.max_iters is not None:
        cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=args.max_iters,
                             val_interval=max(args.max_iters, 1))
        # Keep the warmup (it's iteration-based, by_epoch=False, so it doesn't
        # need epoch info) but drop the epoch-based MultiStepLR decay, which
        # doesn't make sense for a short iter-based confidence check.
        warmup_end = min(500, args.max_iters)
        cfg.param_scheduler = [
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=warmup_end),
        ]

    if args.log_interval is not None:
        cfg.default_hooks.logger = dict(type="LoggerHook", interval=args.log_interval)

    # No custom Dice/IoU validation script yet (see README) -- val disabled for now.
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.test_cfg = None
    cfg.test_dataloader = None
    cfg.test_evaluator = None

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
