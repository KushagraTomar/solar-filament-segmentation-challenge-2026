"""Smoke-test an mmdet config end-to-end on real MAGFiLO data: build the
model + dataloader from the config, then run a couple of real training
iterations (forward, loss, backward, optimizer step) to catch integration
bugs (shape mismatches, registry errors, etc.) before a real training run.

Overrides batch size / image scale / iteration count for a fast, low-memory
check on the local 4GB GPU -- NOT representative of real training speed or
the final input-resolution decision.

Usage:
    python -m src.scripts.smoke_test_config --config configs/pointrend_r50_fpn_magfilo.py
"""
import argparse

from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--work-dir", default="outputs/smoke_test")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir

    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    for step in cfg.train_dataloader.dataset.pipeline:
        if step["type"] == "Resize":
            step["scale"] = (args.crop, args.crop)

    cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=args.max_iters, val_interval=10_000)
    cfg.val_cfg = None
    cfg.val_dataloader = None
    cfg.val_evaluator = None
    cfg.test_cfg = None
    cfg.test_dataloader = None
    cfg.test_evaluator = None
    cfg.param_scheduler = []
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=10_000)
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=1)

    runner = Runner.from_cfg(cfg)
    runner.train()
    print(f"\nSmoke test OK: {args.max_iters} iteration(s) completed without error.")


if __name__ == "__main__":
    main()
