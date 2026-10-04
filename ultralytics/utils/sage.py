# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F


def aligned_iou_xyxy(pred_boxes, target_boxes, eps=1e-7):
    with torch.autocast(device_type=pred_boxes.device.type, enabled=False):
        pred, target = pred_boxes.float(), target_boxes.float()
        pred_size = (pred[..., 2:] - pred[..., :2]).clamp_min(0)
        target_size = (target[..., 2:] - target[..., :2]).clamp_min(0)
        intersection_size = (torch.minimum(pred[..., 2:], target[..., 2:]) - torch.maximum(pred[..., :2], target[..., :2])).clamp_min(0)
        intersection = intersection_size.prod(-1)
        union = pred_size.prod(-1) + target_size.prod(-1) - intersection
        return (intersection / union.clamp_min(eps)).clamp(0, 1)


def sage_geometry(
    pred_boxes,
    target_boxes,
    strides,
    pivot_cells=3.0,
    temperature=0.75,
    stride_floor=1.0,
    shape_weight=0.75,
    charbonnier_eps=0.05,
    quality_gamma=1.0,
    eps=1e-7,
    quality_eps=1e-7,
):
    if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
        raise ValueError("Predicted and target boxes must have matching shapes ending in four coordinates")
    if temperature <= 0 or stride_floor < 0 or charbonnier_eps <= 0 or eps <= 0 or not 0 < quality_eps < 1 or quality_gamma <= 0:
        raise ValueError("Invalid SAGE geometry parameters")
    with torch.autocast(device_type=pred_boxes.device.type, enabled=False):
        pred, target = pred_boxes.float(), target_boxes.float()
        stride = strides.float()
        if stride.ndim == pred.ndim and stride.shape[-1] == 1:
            stride = stride.squeeze(-1)
        stride = torch.broadcast_to(stride, pred.shape[:-1])
        pred_center = 0.5 * (pred[..., :2] + pred[..., 2:])
        target_center = 0.5 * (target[..., :2] + target[..., 2:])
        pred_size = (pred[..., 2:] - pred[..., :2]).clamp_min(0)
        target_size = (target[..., 2:] - target[..., :2]).clamp_min(0)
        side = torch.sqrt(target_size.prod(-1) + eps)
        grid_size = side / (stride + eps)
        alpha = torch.sigmoid((grid_size - pivot_cells) / temperature)
        center_denominator = (side + stride_floor * stride).clamp_min(eps)
        size_denominator = (target_size + stride_floor * stride.unsqueeze(-1)).clamp_min(eps)
        center_residual = (pred_center - target_center) / center_denominator.unsqueeze(-1)
        size_residual = (pred_size - target_size) / size_denominator
        center_energy = (torch.sqrt(center_residual.square() + charbonnier_eps**2) - charbonnier_eps).mean(-1)
        shape_energy = (torch.sqrt(size_residual.square() + charbonnier_eps**2) - charbonnier_eps).mean(-1)
        robust = center_energy + shape_weight * shape_energy
        iou = aligned_iou_xyxy(pred, target, eps)
        loss = alpha * (1.0 - iou) + (1.0 - alpha) * robust
        quality_energy = alpha * (-torch.log(iou.clamp_min(quality_eps))) + (1.0 - alpha) * robust
        quality = torch.exp(-quality_gamma * quality_energy).clamp(0, 1).detach()
        return {
            "loss": loss,
            "quality": quality,
            "iou": iou,
            "alpha_iou": alpha,
            "grid_size": grid_size,
            "center": center_energy,
            "shape": shape_energy,
            "robust_geometry": robust,
            "quality_energy": quality_energy,
        }


class SAGEGeometry(nn.Module):
    def __init__(
        self,
        pivot_cells=3.0,
        temperature=0.75,
        stride_floor=1.0,
        shape_weight=0.75,
        charbonnier_eps=0.05,
        quality_gamma=1.0,
        eps=1e-7,
        quality_eps=1e-7,
    ):
        super().__init__()
        self.pivot_cells = float(pivot_cells)
        self.temperature = float(temperature)
        self.stride_floor = float(stride_floor)
        self.shape_weight = float(shape_weight)
        self.charbonnier_eps = float(charbonnier_eps)
        self.quality_gamma = float(quality_gamma)
        self.eps = float(eps)
        self.quality_eps = float(quality_eps)

    def forward(self, pred_boxes, target_boxes, strides):
        return sage_geometry(
            pred_boxes,
            target_boxes,
            strides,
            self.pivot_cells,
            self.temperature,
            self.stride_floor,
            self.shape_weight,
            self.charbonnier_eps,
            self.quality_gamma,
            self.eps,
            self.quality_eps,
        )


def weighted_sage_losses(geometry, positive_quality_logits, negative_quality_logits, positive_weights, negative_weight=0.05):
    with torch.autocast(device_type=positive_quality_logits.device.type, enabled=False):
        weights = positive_weights.float().reshape(-1).detach()
        losses = geometry["loss"].float().reshape(-1)
        targets = geometry["quality"].float().reshape(-1).detach()
        logits = positive_quality_logits.float().reshape(-1)
        negatives = negative_quality_logits.float().reshape(-1)
        if not (weights.numel() == losses.numel() == targets.numel() == logits.numel()):
            raise ValueError("Positive losses, weights, targets, and logits must have equal lengths")
        denominator = weights.sum().clamp_min(1.0)
        box = (weights * losses).sum() / denominator
        quality_positive = (weights * F.binary_cross_entropy_with_logits(logits, targets, reduction="none")).sum() / denominator
        quality_negative = F.binary_cross_entropy_with_logits(negatives, torch.zeros_like(negatives), reduction="sum") / max(negatives.numel(), 1)
        return {
            "box": box,
            "quality": quality_positive + negative_weight * quality_negative,
            "quality_positive": quality_positive,
            "quality_negative": quality_negative,
        }
