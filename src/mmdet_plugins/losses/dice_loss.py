"""Dice loss usable as `loss_mask`/`loss_point` on mmdet's mask/point heads.

mmdet's mask/point heads always invoke `loss_mask(pred, target, label)` with
a positional `label` argument -- the per-instance class index, used by
`CrossEntropyLoss(use_mask=True)` to select which of the (up to num_classes)
predicted channels to read a given instance's mask from (see
fcn_mask_head.py / mask_point_head.py `loss_and_target`). Under
`class_agnostic=True` there's supposed to be only one channel, so `label` is
always `torch.zeros_like(pos_labels)` -- but `CoarseMaskHead` (used by
PointRend) has an upstream bug where its own fc_logits layer hardcodes
`self.num_classes` output channels regardless of `class_agnostic`, so its
predictions stay multi-channel either way (verified against the installed
mmdet 3.3.0 source). `MaskPointHead` does not have this bug.

mmdet's own `DiceLoss` reuses that same positional slot as a per-sample
*weight* instead (`forward(pred, target, weight=None, ...)`) and multiplies
the loss by it -- under class_agnostic that weight is all zeros, silently
zeroing the entire mask loss with no error. This wrapper instead treats
`label` the way `CrossEntropyLoss(use_mask=True)` does: if there's more than
one channel, select the channel matching each instance's class before
computing Dice; this is correct regardless of whether a given head's
class_agnostic flag actually collapsed it to one channel or not.

Flattens all dims after channel-selection (`.flatten(1)`) rather than
assuming a fixed number of spatial dims, since it's used both by
CoarseMaskHead (pred shape (N, C, H, W)) and MaskPointHead (pred shape
(N, C, num_points)).
"""
import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module()
class MagfiloDiceLoss(nn.Module):
    def __init__(self, loss_weight=1.0, eps=1e-3):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target, label=None, reduction_override=None, avg_factor=None):
        if pred.size(1) > 1:
            inds = torch.arange(pred.size(0), device=pred.device)
            pred = pred[inds, label]
        else:
            pred = pred.squeeze(1)

        prob = pred.sigmoid().flatten(1)
        target = target.flatten(1).float()

        intersection = (prob * target).sum(dim=1)
        denom = (prob * prob).sum(dim=1) + (target * target).sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        loss = 1.0 - dice

        reduction = reduction_override or "mean"
        if reduction == "mean":
            loss = loss.sum() / avg_factor if avg_factor else loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss
