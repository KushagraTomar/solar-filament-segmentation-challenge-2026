"""Runtime patch for a real upstream mmdet bug: `CoarseMaskHead` (used by
PointRend) hardcodes its `fc_logits` output to `num_classes` channels
regardless of `class_agnostic=True` (see src/mmdet_plugins/losses/dice_loss.py
for the training-time consequence and how MagfiloDiceLoss works around it
there). `MaskPointHead`'s OWN output correctly respects `class_agnostic=True`
(1 channel) -- but its `fc_in_channels = in_channels + num_classes` sizing is
unconditional, so its trained fc layers still expect the *input* coarse
features at the full num_classes width regardless of class_agnostic. So the
as-trained checkpoint has: CoarseMaskHead output = 4 channels (real bug),
MaskPointHead output = 1 channel (correct), MaskPointHead's fc layers built
expecting 4-channel coarse input (unrelated to the bug, just how the layer is
always sized).

This mismatch breaks mmdet's own `PointRendRoIHead._mask_point_forward_test`
in two ways if left unpatched:
  1. `coarse_point_feats` must stay 4-channel (matches trained fc layer input
     size) -- reducing it early breaks `MaskPointHead.forward()`.
  2. The final `scatter_` naively broadcasts `mask_point_pred` (1 channel)
     into all 4 channels of `refined_mask_pred`, assuming matching channel
     counts -- shape-mismatch RuntimeError.
  3. `_predict_by_feat_single` skips its own per-instance class-channel
     selection when `class_agnostic=True` (assuming the tensor is already
     reduced to 1 channel by then) -- so this function must return an
     already-reduced (N, 1, H, W) tensor, not the raw 4-channel one.

Fix: keep `mask_preds`/`coarse_point_feats` at their real (4-channel) width
throughout the refinement loop (matching trained layer shapes), scatter the
single refined point value into only the per-instance predicted-class
channel (not broadcast into all four), and reduce to that one selected
channel only once, right before returning.

This is inference-only and doesn't touch any trained weights -- retraining
with a from-scratch fixed CoarseMaskHead would need different fc_logits
weight shapes, not just a fix at load time.

Call `apply_pointrend_test_patch()` once before running inference.
"""
import torch
import torch.nn.functional as F
from mmcv.ops import point_sample
from mmdet.models.roi_heads.point_rend_roi_head import PointRendRoIHead


def _patched_mask_point_forward_test(self, x, rois, label_preds, mask_preds):
    refined_mask_pred = mask_preds.clone()
    inds = torch.arange(mask_preds.size(0), device=mask_preds.device)

    for subdivision_step in range(self.test_cfg.subdivision_steps):
        refined_mask_pred = F.interpolate(
            refined_mask_pred,
            scale_factor=self.test_cfg.scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        num_rois, channels, mask_height, mask_width = refined_mask_pred.shape
        if (
            self.test_cfg.subdivision_num_points
            >= self.test_cfg.scale_factor**2 * mask_height * mask_width
            and subdivision_step < self.test_cfg.subdivision_steps - 1
        ):
            continue
        point_indices, rel_roi_points = self.point_head.get_roi_rel_points_test(
            refined_mask_pred, label_preds, cfg=self.test_cfg
        )
        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x=x, rois=rois, rel_roi_points=rel_roi_points
        )
        # Full-channel coarse features -- matches MaskPointHead's trained
        # fc layer input size (fc_in_channels = in_channels + num_classes,
        # unconditional on class_agnostic).
        coarse_point_feats = point_sample(mask_preds, rel_roi_points)
        mask_point_pred = self.point_head(fine_grained_point_feats, coarse_point_feats)

        refined_mask_pred = refined_mask_pred.reshape(num_rois, channels, mask_height * mask_width)
        if mask_point_pred.size(1) == 1 and channels > 1:
            # Point head is class-agnostic (one refined value per point);
            # write it into just the predicted-class channel per instance,
            # not all `channels` (which naive broadcasting would require
            # mask_point_pred to already match).
            selected = refined_mask_pred[inds, label_preds]
            selected = selected.scatter(1, point_indices, mask_point_pred.squeeze(1))
            refined_mask_pred[inds, label_preds] = selected
        else:
            point_indices_expanded = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_mask_pred = refined_mask_pred.scatter_(2, point_indices_expanded, mask_point_pred)
        refined_mask_pred = refined_mask_pred.view(num_rois, channels, mask_height, mask_width)

    # `_predict_by_feat_single` skips its own per-instance class-channel
    # selection when `class_agnostic=True` (assumes the tensor is already
    # reduced by now) -- so reduce here, once, at the end.
    if refined_mask_pred.size(1) > 1:
        refined_mask_pred = refined_mask_pred[inds, label_preds].unsqueeze(1)

    return refined_mask_pred


def apply_pointrend_test_patch():
    PointRendRoIHead._mask_point_forward_test = _patched_mask_point_forward_test
