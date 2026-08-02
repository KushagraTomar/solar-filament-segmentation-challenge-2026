"""Dice + clDice loss for filament mask supervision.

clDice (Shit et al., "clDice: A Novel Topology-Preserving Loss Function for
Tubular Structure Segmentation", CVPR 2021) measures overlap between each
mask's centerline (skeleton) and the other mask's full region, which rewards
connectivity/completeness independent of a structure's local thickness --
exactly the failure mode Dice misses for thin barbs and small gaps.

MAGFiLO ships a hand-annotated spine per filament (annotation["spine"]). When
a rasterized spine is passed as `gt_skeleton`, it is used as the ground-truth
centerline directly, instead of approximating it via soft-skeletonization of
the mask -- see coco_utils.rasterize_spine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iterations=10):
    """Differentiable morphological skeleton approximation (iterative soft
    erosion + tophat). `img` is a (B, 1, H, W) tensor with values in [0, 1]."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iterations):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_dice_loss(y_pred, y_true, smooth=1.0):
    intersection = (y_pred * y_true).sum(dim=(-2, -1))
    denom = y_pred.sum(dim=(-2, -1)) + y_true.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def soft_cldice_loss(y_pred, y_true, iterations=10, smooth=1.0, gt_skeleton=None):
    """`gt_skeleton`, if given, replaces soft-skeletonization of `y_true` with
    a precomputed ground-truth centerline (e.g. the rasterized MAGFiLO spine)."""
    skel_pred = soft_skeletonize(y_pred, iterations)
    skel_true = gt_skeleton if gt_skeleton is not None else soft_skeletonize(y_true, iterations)

    tprec = (skel_pred * y_true).sum(dim=(-2, -1)) / (skel_pred.sum(dim=(-2, -1)) + smooth)
    tsens = (skel_true * y_pred).sum(dim=(-2, -1)) / (skel_true.sum(dim=(-2, -1)) + smooth)

    # tprec/tsens are already normalized to [0, 1] by the `smooth` term above;
    # this harmonic-mean combination only needs a tiny epsilon to guard 0/0,
    # not `smooth` again (which would wrongly deflate the score toward 0.5).
    eps = 1e-8
    cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + eps)
    return 1.0 - cl_dice.mean()


class DiceClDiceLoss(nn.Module):
    """Combined loss: (1 - alpha) * dice + alpha * clDice."""

    def __init__(self, iterations=10, alpha=0.3, smooth=1.0):
        super().__init__()
        self.iterations = iterations
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, y_pred, y_true, gt_skeleton=None):
        """y_pred: (B, 1, H, W) sigmoid probabilities. y_true: (B, 1, H, W) binary.
        gt_skeleton: optional (B, 1, H, W) rasterized ground-truth spine."""
        dice = soft_dice_loss(y_pred, y_true, self.smooth)
        cldice = soft_cldice_loss(y_pred, y_true, self.iterations, self.smooth, gt_skeleton)
        return (1.0 - self.alpha) * dice + self.alpha * cldice
