# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.sage import SAGEGeometry
from ultralytics.utils.tal import make_anchors


class AeroGuardDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model, tal_topk=10)
        config = dict(model.yaml.get("aeroguard", {}))
        for key in ("use_sage", "quality_supervision", "quality_target", "quality_gain", "quality_negative_weight"):
            value = getattr(model.args, key, None)
            if value is not None:
                config[key] = value
        self.use_sage = bool(config.get("use_sage", False))
        self.has_quality = bool(getattr(model.model[-1], "has_quality", False))
        self.quality_supervision = bool(config.get("quality_supervision", False))
        self.quality_target = config.get("quality_target", "composite")
        self.quality_gain = float(config.get("quality_gain", 0.5))
        self.quality_negative_weight = float(config.get("quality_negative_weight", 0.05))
        if self.quality_supervision and not self.has_quality:
            raise ValueError("Quality supervision requires Detect_SJRH")
        if self.quality_target not in {"composite", "iou"}:
            raise ValueError("quality_target must be composite or iou")
        self.geometry = SAGEGeometry(**config.get("sage", {}))
        self.no = self.reg_max * 4 + self.nc + int(self.has_quality)
        self.last_diagnostics = {}

    def __call__(self, preds, batch):
        feats = preds[1] if isinstance(preds, tuple) else preds
        batch_size = feats[0].shape[0]
        merged = torch.cat([x.view(batch_size, self.no, -1) for x in feats], 2)
        sizes = (self.reg_max * 4, self.nc, 1) if self.has_quality else (self.reg_max * 4, self.nc)
        parts = merged.split(sizes, 1)
        with torch.autocast(device_type=feats[0].device.type, enabled=False):
            distribution = parts[0].float().permute(0, 2, 1).contiguous()
            scores = parts[1].float().permute(0, 2, 1).contiguous()
            quality = parts[2].float().permute(0, 2, 1).squeeze(-1) if self.has_quality else None
            anchors, stride = make_anchors(feats, self.stride, 0.5)
            anchors, stride = anchors.float(), stride.float()
            image_size = torch.tensor(feats[0].shape[2:], device=self.device, dtype=torch.float32) * self.stride[0]
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device).float(), batch_size, image_size[[1, 0, 1, 0]])
            labels, gt_boxes = targets.split((1, 4), 2)
            mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0)
            predicted = self.bbox_decode(anchors, distribution)
            _, target_boxes, target_scores, foreground, _ = self.assigner(
                scores.detach().sigmoid(),
                (predicted.detach() * stride).to(gt_boxes.dtype),
                anchors * stride,
                labels,
                gt_boxes,
                mask_gt,
            )
            foreground = foreground.bool()
            target_scores = target_scores.float().detach()
            normalizer = target_scores.sum().clamp_min(1.0)
            zero = distribution.sum() * 0.0
            box_loss, dfl_loss = zero, zero
            cls_loss = F.binary_cross_entropy_with_logits(scores, target_scores, reduction="sum") / normalizer
            positive_quality = quality.sum() * 0.0 if quality is not None else zero
            negative_quality = positive_quality
            count = int(foreground.sum().detach())
            if count:
                grid_targets = target_boxes / stride
                ciou_loss, dfl_loss = self.bbox_loss(
                    distribution, predicted, anchors, grid_targets, target_scores, normalizer, foreground
                )
                weights = target_scores.sum(-1)[foreground]
                per_prediction_stride = stride.unsqueeze(0).expand(batch_size, -1, -1)[foreground]
                geometry = self.geometry(
                    (predicted * stride)[foreground], target_boxes[foreground], per_prediction_stride
                )
                box_loss = (weights * geometry["loss"]).sum() / normalizer if self.use_sage else ciou_loss
                if self.has_quality and self.quality_supervision:
                    target = geometry["quality"] if self.quality_target == "composite" else geometry["iou"].detach()
                    positive_quality = (
                        weights * F.binary_cross_entropy_with_logits(quality[foreground], target.detach(), reduction="none")
                    ).sum() / normalizer
            if self.has_quality and self.quality_supervision:
                negatives = quality[~foreground]
                negative_quality = F.binary_cross_entropy_with_logits(
                    negatives, torch.zeros_like(negatives), reduction="sum"
                ) / max(negatives.numel(), 1)
            quality_loss = positive_quality + self.quality_negative_weight * negative_quality
            loss = torch.stack((
                box_loss * self.hyp.box,
                cls_loss * self.hyp.cls,
                dfl_loss * self.hyp.dfl,
                quality_loss * self.quality_gain * self.hyp.box,
            ))
            self.last_diagnostics = {
                "positive_count": count,
                "negative_count": int((~foreground).sum().detach()),
                "positive_weight_sum": float(target_scores.sum().detach()),
                "quality_positive": float(positive_quality.detach()),
                "quality_negative": float(negative_quality.detach()),
                "box_unscaled": float(box_loss.detach()),
            }
        return loss.sum() * batch_size, loss.detach()

