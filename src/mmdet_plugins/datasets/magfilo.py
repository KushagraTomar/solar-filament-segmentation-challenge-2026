"""Custom mmdet dataset for MAGFiLO.

Registers the 4 MAGFiLO categories and preserves the per-annotation `spine`
field (not part of standard COCO) through mmdet's data-loading pipeline, so a
downstream pipeline transform can rasterize it into a ground-truth skeleton
for the clDice loss.

mmdet's base `CocoDataset.parse_data_info` assigns
`instance['mask'] = ann['segmentation']` by reference (not a copy), so we can
recover which raw annotation produced each parsed instance via object
identity, without re-implementing mmdet's internal filtering logic (bbox/area
validity, category membership, etc.) -- that keeps this override correct
across mmdet versions instead of drifting out of sync with a duplicated copy
of that logic.
"""
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class MagfiloDataset(CocoDataset):
    METAINFO = {
        "classes": ("Left", "Right", "Unidentifiable", "Ambiguous"),
        "palette": [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)],
    }

    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)

        ann_info = raw_data_info["raw_ann_info"]
        spine_by_seg_id = {id(ann["segmentation"]): ann.get("spine") for ann in ann_info}

        for instance in data_info["instances"]:
            instance["spine"] = spine_by_seg_id.get(id(instance.get("mask")))

        return data_info
