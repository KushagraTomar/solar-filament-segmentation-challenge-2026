"""PointRend (Mask R-CNN + point-based mask refinement) for MAGFiLO filament
instance segmentation.

Architecture note: mmdet ships PointRend as its own RoIHead
(`PointRendRoIHead`, single-stage Mask R-CNN), not as a mask-head type
pluggable into `HybridTaskCascadeRoIHead`'s cascade. Combining Cascade R-CNN +
HTC's semantic branch + PointRend as one architecture would require writing a
custom RoIHead -- deferred; this config uses PointRend only (chosen over
Cascade+HTC given the fine-boundary/barb recovery is the more direct fit for
this competition's challenges).

Losses: `loss_mask` / `loss_point` use `MagfiloDiceLoss` (src/mmdet_plugins/
losses/dice_loss.py) instead of the default CrossEntropyLoss, so mask
supervision is closer to the eval metric (Dice/IoU). NOT mmdet's own built-in
`DiceLoss` -- that class treats the mask/point heads' positional `label`
argument as a per-sample weight, which is all-zeros under class_agnostic=True
and would silently zero out the entire loss; MagfiloDiceLoss just ignores it.

The custom spine-aware clDice loss (src/losses/cldice.py) is NOT yet wired in
here -- CoarseMaskHead/MaskPointHead operate on per-RoI cropped/resized
targets, not full-image masks, so clDice's ground-truth skeleton needs the
same per-RoI cropping applied to the rasterized spine before it can be used
inside these heads. That's the next integration step.

Input resolution: 1024x1024 (keep_ratio) is a placeholder. The tiling-vs-resize
strategy for MAGFiLO's native 2048x2048 images is still an open decision
(flagged during architecture design) -- revisit before a real training run.

This config targets a cloud GPU (batch_size=2 below). For a local smoke test
on the 4GB GTX 1650, override batch_size=1 and use a smaller crop -- see
src/scripts/smoke_test_config.py.
"""
custom_imports = dict(imports=["src.mmdet_plugins"], allow_failed_imports=False)

dataset_type = "MagfiloDataset"
data_root = "data/raw/MAGFiLO_1.0_Kaggle_2026/train/"
backend_args = None

num_classes = 4

model = dict(
    type="PointRend",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="caffe",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://detectron2/resnet50_caffe"),
    ),
    neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="PointRendRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
        mask_roi_extractor=dict(
            type="GenericRoIExtractor",
            aggregation="concat",
            roi_layer=dict(type="SimpleRoIAlign", output_size=14),
            out_channels=256,
            featmap_strides=[4],
        ),
        mask_head=dict(
            type="CoarseMaskHead",
            num_fcs=2,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=num_classes,
            class_agnostic=True,
            loss_mask=dict(type="MagfiloDiceLoss", loss_weight=1.0),
        ),
        point_head=dict(
            type="MaskPointHead",
            num_fcs=3,
            in_channels=256,
            fc_channels=256,
            num_classes=num_classes,
            class_agnostic=True,
            coarse_pred_each_layer=True,
            loss_point=dict(type="MagfiloDiceLoss", loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=7,
            num_points=14 * 14,
            oversample_ratio=3,
            importance_sample_ratio=0.75,
            pos_weight=-1,
            debug=False,
        ),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
            subdivision_steps=5,
            subdivision_num_points=28 * 28,
            scale_factor=2,
        ),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="../../../../data/splits/train_coco.json",
        data_prefix=dict(img="train_images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="../../../../data/splits/val_coco.json",
        data_prefix=dict(img="train_images/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# mmdet's built-in CocoMetric drives pycocotools COCOeval, which is not
# verified safe against MAGFiLO's string image/annotation ids (see
# src/data/coco_utils.py docstring) and doesn't compute the competition's
# actual rubric (Dice score distributions, one-to-many/many-to-one matching)
# anyway. Validation uses a custom Dice/IoU script instead -- not yet written.
val_evaluator = None
test_evaluator = None
val_cfg = None
test_cfg = None

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001),
)

default_scope = "mmdet"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
