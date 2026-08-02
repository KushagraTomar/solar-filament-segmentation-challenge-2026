"""Builds the full-image binary semantic-segmentation ground truth (filament
vs. background) that HTC's FusedSemanticHead needs, since MAGFiLO ships no
pre-rendered semantic PNGs (unlike COCO-Stuff, which HTC's stock pipeline
expects via `LoadAnnotations(with_seg=True)` + a `seg_map_path` file).

Must run AFTER Resize/RandomFlip in the pipeline so the merged mask lines up
with the augmented image, since it reads `results['gt_masks']` (already
transformed by that point) rather than the raw annotation polygons.

clDice/spine-aware supervision for this branch is a follow-up increment (needs
subclassing FusedSemanticHead + HybridTaskCascadeRoIHead to thread an extra
skeleton tensor through their fixed loss(...) signatures) -- this transform
only builds the plain binary label mask, consumed by mmdet's stock
CrossEntropyLoss-based semantic head for now.
"""
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BuildSemanticGT(BaseTransform):
    def transform(self, results):
        gt_masks = results["gt_masks"]
        h, w = results["img_shape"][:2]

        if len(gt_masks) == 0:
            seg_map = np.zeros((h, w), dtype=np.uint8)
        else:
            if hasattr(gt_masks, "to_ndarray"):
                masks = gt_masks.to_ndarray()
            else:
                masks = gt_masks.masks
            seg_map = (masks.any(axis=0)).astype(np.uint8)

        results["gt_seg_map"] = seg_map
        return results
