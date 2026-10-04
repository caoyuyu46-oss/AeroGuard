# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(c_in, c_out, kernel_size=1, groups=1, dilation=1):
    layer = nn.Conv2d(
        c_in,
        c_out,
        kernel_size,
        padding=dilation * (kernel_size - 1) // 2,
        groups=groups,
        dilation=dilation,
        bias=True,
    )
    nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
    nn.init.zeros_(layer.bias)
    return layer


def _average_replicate(x, kernel_size):
    padding = kernel_size // 2
    return F.avg_pool2d(F.pad(x, (padding,) * 4, mode="replicate"), kernel_size, stride=1)


def _area_resize(x, size):
    if torch.onnx.is_in_onnx_export():
        height, width = int(x.shape[-2]), int(x.shape[-1])
        target_height, target_width = int(size[0]), int(size[1])
        if height % target_height or width % target_width:
            raise ValueError("Fixed ONNX area resampling requires integer spatial reduction factors")
        factor = (height // target_height, width // target_width)
        return F.avg_pool2d(x, kernel_size=factor, stride=factor)
    return F.interpolate(x, size=size, mode="area")


class FixedBinomialBlur(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0
        self.register_buffer("kernel", kernel.reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.channels = int(channels)

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.kernel.to(x.dtype), groups=self.channels)


class SharedDetailExtractor(nn.Module):
    def __init__(self, c_in, channels=64, tau_init=0.80, sharpness=4.0, eps=1e-6, use_shrink=True):
        super().__init__()
        if tau_init <= 0 or sharpness <= 0 or eps <= 0:
            raise ValueError("Threshold, sharpness, and stabilizer must be positive")
        self.channels = int(channels)
        self.eps = float(eps)
        self.sharpness = float(sharpness)
        self.use_shrink = bool(use_shrink)
        self.shallow_proj = _conv(int(c_in), self.channels)
        self.blur = FixedBinomialBlur(self.channels)
        self.tau_raw = nn.Parameter(torch.full((1, self.channels, 1, 1), math.log(math.expm1(tau_init))))

    def _forward_impl(self, x):
        projected = self.shallow_proj(x)
        low = self.blur(projected)
        detail = projected - low
        with torch.autocast(device_type=x.device.type, enabled=False):
            rms = torch.sqrt(_average_replicate(detail.float().square(), 3) + self.eps)
            normalized = detail.float() / (rms + self.eps)
            threshold = F.softplus(self.tau_raw.float())
            shrinkage = torch.sigmoid(self.sharpness * (normalized.abs() - threshold))
            if not self.use_shrink:
                shrinkage = torch.ones_like(shrinkage)
            candidates = (detail.float() * shrinkage).to(detail.dtype)
        return (candidates, low), {
            "projected": projected,
            "low": low,
            "detail": detail,
            "rms": rms,
            "normalized": normalized,
            "shrinkage": shrinkage,
            "candidates": candidates,
        }

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)


class RDET(nn.Module):
    def __init__(
        self,
        c_receiver,
        channels=64,
        gate_hidden=32,
        transfer_cap=0.50,
        transfer_init=0.10,
        eps=1e-6,
        use_gate=True,
        conditioning_mode="full",
    ):
        super().__init__()
        if transfer_cap <= 0 or abs(transfer_init) >= transfer_cap or eps <= 0:
            raise ValueError("Transfer initialization must lie strictly inside a positive transfer bound")
        self.channels = int(channels)
        self.eps = float(eps)
        self.transfer_cap = float(transfer_cap)
        self.use_gate = bool(use_gate)
        if conditioning_mode not in {"full", "no_agreement", "no_salience"}:
            raise ValueError("Unknown receiver conditioning mode")
        self.conditioning_mode = conditioning_mode
        self.receiver_proj = _conv(int(c_receiver), self.channels)
        self.gate = nn.Sequential(_conv(3, gate_hidden), nn.SiLU(), _conv(gate_hidden, 1))
        self.transfer = nn.Sequential(
            _conv(self.channels, self.channels, 3, groups=self.channels),
            nn.SiLU(),
            _conv(self.channels, int(c_receiver)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha_raw = nn.Parameter(torch.tensor(math.atanh(transfer_init / transfer_cap)))

    def _forward_impl(self, x):
        shared, receiver = x
        candidates, low = shared
        size = receiver.shape[-2:]
        candidates = _area_resize(candidates, size)
        low = _area_resize(low, size)
        projected = self.receiver_proj(receiver)
        with torch.autocast(device_type=receiver.device.type, enabled=False):
            e, s, q = candidates.float(), low.float(), projected.float()
            strength = e.abs().mean(1, keepdim=True) / (s.abs().mean(1, keepdim=True) + self.eps)
            numerator = (s * q).sum(1, keepdim=True)
            denominator = torch.linalg.vector_norm(s, dim=1, keepdim=True) * torch.linalg.vector_norm(q, dim=1, keepdim=True)
            agreement = 0.5 * (1.0 + numerator / (denominator + self.eps))
            salience = q.abs().mean(1, keepdim=True)
            salience = salience / (_average_replicate(salience, 5) + self.eps)
            if self.conditioning_mode == "no_agreement":
                agreement = torch.zeros_like(agreement)
            if self.conditioning_mode == "no_salience":
                salience = torch.zeros_like(salience)
            conditioning = torch.cat((strength, agreement, salience), 1).to(projected.dtype)
        gate = torch.sigmoid(self.gate(conditioning)) if self.use_gate else torch.ones_like(strength).to(projected.dtype)
        transformed = self.transfer(candidates)
        alpha = self.transfer_cap * torch.tanh(self.alpha_raw)
        increment = alpha * gate * transformed
        return receiver + increment, {
            "candidate": candidates,
            "low": low,
            "receiver_projection": projected,
            "conditioning": conditioning,
            "gate": gate,
            "transformed": transformed,
            "alpha": alpha,
            "increment": increment,
        }

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)


class BDRP(nn.Module):
    def __init__(self, c_inputs, channels=64, rho=0.35, eps=1e-6, use_gate=True, use_bound=True, reference_mode="local"):
        super().__init__()
        if len(c_inputs) != 3 or rho < 0 or eps <= 0:
            raise ValueError("BDRP requires three inputs, a nonnegative bound, and a positive stabilizer")
        self.channels = int(channels)
        self.rho = float(rho)
        self.eps = float(eps)
        self.use_gate = bool(use_gate)
        self.use_bound = bool(use_bound)
        if reference_mode not in {"local", "global"}:
            raise ValueError("Unknown fusion amplitude reference")
        self.reference_mode = reference_mode
        self.p2_proj = _conv(int(c_inputs[0]) * 4, self.channels)
        self.topdown_proj = _conv(int(c_inputs[1]), self.channels)
        self.lateral_proj = _conv(int(c_inputs[2]), self.channels)
        self.base_fuse = nn.Sequential(
            _conv(self.channels * 3, self.channels),
            nn.SiLU(),
            _conv(self.channels, self.channels, 3),
            nn.SiLU(),
        )
        self.blur = FixedBinomialBlur(self.channels)
        self.dw1 = _conv(self.channels, self.channels, 3, groups=self.channels)
        self.dw2 = _conv(self.channels, self.channels, 3, groups=self.channels, dilation=2)
        self.mix = _conv(self.channels * 2, self.channels)
        self.gate = nn.Sequential(_conv(3, 32), nn.SiLU(), _conv(32, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _forward_impl(self, x):
        p2, topdown, lateral = x
        h, w = lateral.shape[-2:]
        if p2.shape[-2:] != (h * 2, w * 2):
            raise ValueError("BDRP P2 dimensions must be exactly twice the lateral P3 dimensions")
        a2 = self.p2_proj(F.pixel_unshuffle(p2, 2))
        at = self.topdown_proj(F.interpolate(topdown, size=(h, w), mode="nearest"))
        a3 = self.lateral_proj(lateral)
        base = self.base_fuse(torch.cat((a2, at, a3), 1))
        low = self.blur(base)
        detail = base - low
        with torch.autocast(device_type=base.device.type, enabled=False):
            a2f, atf, a3f = a2.float(), at.float(), a3.float()
            strength = detail.float().abs().mean(1, keepdim=True) / (low.float().abs().mean(1, keepdim=True) + self.eps)
            topdown_difference = (atf - a3f).abs().mean(1, keepdim=True) / (
                0.5 * (atf.abs().mean(1, keepdim=True) + a3f.abs().mean(1, keepdim=True)) + self.eps
            )
            shallow_difference = (a2f - a3f).abs().mean(1, keepdim=True) / (
                0.5 * (a2f.abs().mean(1, keepdim=True) + a3f.abs().mean(1, keepdim=True)) + self.eps
            )
            conditioning = torch.cat((strength, topdown_difference, shallow_difference), 1).to(base.dtype)
            dimensions = 1 if self.reference_mode == "local" else (1, 2, 3)
            reference = torch.sqrt(base.float().square().mean(dimensions, keepdim=True) + self.eps)
        gate = torch.sigmoid(self.gate(conditioning)) if self.use_gate else torch.ones_like(strength).to(base.dtype)
        raw = self.mix(torch.cat((self.dw1(detail), self.dw2(detail)), 1))
        with torch.autocast(device_type=base.device.type, enabled=False):
            bounded = self.rho * reference * torch.tanh(raw.float() / (reference + self.eps)) if self.use_bound else raw.float()
            increment = gate.float() * bounded
            output = (base.float() + increment).to(base.dtype)
        return output, {
            "base": base,
            "low": low,
            "detail": detail,
            "conditioning": conditioning,
            "gate": gate,
            "raw_delta": raw,
            "bounded_delta": bounded,
            "reference_rms": reference,
            "increment": increment,
        }

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)


class TaskBoundedSampler(nn.Module):
    def __init__(self, channels=96, max_offset_cells=1.50, hidden=24):
        super().__init__()
        if max_offset_cells < 0:
            raise ValueError("The per-axis displacement bound must be nonnegative")
        self.max_offset_cells = float(max_offset_cells)
        self.predictor = nn.Sequential(_conv(channels, hidden, 3), nn.SiLU(), _conv(hidden, 3))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def _forward_impl(self, x):
        raw = self.predictor(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            raw = raw.float()
            radius = self.max_offset_cells * torch.sigmoid(raw[:, 2:3])
            offset = radius * torch.tanh(raw[:, :2])
            batch, _, height, width = x.shape
            yy = torch.linspace(-1, 1, height, device=x.device, dtype=torch.float32) if height > 1 else torch.zeros(1, device=x.device)
            xx = torch.linspace(-1, 1, width, device=x.device, dtype=torch.float32) if width > 1 else torch.zeros(1, device=x.device)
            gy, gx = torch.meshgrid(yy, xx, indexing="ij")
            base = torch.stack((gx, gy), -1).unsqueeze(0).expand(batch, -1, -1, -1)
            normalized = torch.stack((2.0 * offset[:, 0] / max(width - 1, 1), 2.0 * offset[:, 1] / max(height - 1, 1)), -1)
            grid = base + normalized
            sampled = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=True).to(x.dtype)
        return sampled, {"offset_cells": offset, "radius_cells": radius, "grid": grid}

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)


class SJRHLevel(nn.Module):
    def __init__(self, c_in, nc, reg_max=16, hidden=96, max_offset_cells=1.50, shared_sampling=False):
        super().__init__()
        self.stem = nn.Sequential(
            _conv(c_in, hidden),
            nn.SiLU(),
            _conv(hidden, hidden, 3, groups=hidden),
            nn.SiLU(),
            _conv(hidden, hidden),
            nn.SiLU(),
        )
        self.cls_sampler = TaskBoundedSampler(hidden, max_offset_cells, hidden=24)
        self.reg_sampler = self.cls_sampler if shared_sampling else TaskBoundedSampler(hidden, max_offset_cells, hidden=24)
        self.cls_refine = nn.Sequential(_conv(hidden, hidden, 3, groups=hidden), nn.SiLU(), _conv(hidden, hidden))
        self.reg_refine = nn.Sequential(_conv(hidden, hidden, 3, groups=hidden), nn.SiLU(), _conv(hidden, hidden))
        self.box_pred = _conv(hidden, 4 * reg_max)
        self.cls_pred = _conv(hidden, nc)
        self.quality_pred = nn.Sequential(_conv(2 * hidden, 32), nn.SiLU(), _conv(32, 1))

    def _forward_impl(self, x):
        shared = self.stem(x)
        cls_sample, cls_meta = self.cls_sampler.forward_with_diagnostics(shared)
        if self.reg_sampler is self.cls_sampler:
            reg_sample, reg_meta = cls_sample, cls_meta
        else:
            reg_sample, reg_meta = self.reg_sampler.forward_with_diagnostics(shared)
        cls_feature = self.cls_refine(cls_sample)
        reg_feature = self.reg_refine(reg_sample)
        difference = (cls_feature - reg_feature).abs()
        values = (
            self.box_pred(reg_feature),
            self.cls_pred(cls_feature),
            self.quality_pred(torch.cat((shared, difference), 1)),
        )
        return values, {
            "shared": shared,
            "cls_feature": cls_feature,
            "reg_feature": reg_feature,
            "cls_sampling": cls_meta,
            "reg_sampling": reg_meta,
        }

    def forward(self, x):
        return self._forward_impl(x)[0]

    def forward_with_diagnostics(self, x):
        return self._forward_impl(x)


class SJRHCore(nn.Module):
    def __init__(self, nc, ch, reg_max=16, hidden=96, max_offset_cells=1.50, shared_sampling=False):
        super().__init__()
        self.levels = nn.ModuleList(
            SJRHLevel(c, nc, reg_max, hidden, max_offset_cells, shared_sampling) for c in ch
        )

    def forward(self, x):
        if len(x) != len(self.levels):
            raise ValueError("The feature count must match the number of SJRH levels")
        return [level(feature) for level, feature in zip(self.levels, x)]

    def forward_with_diagnostics(self, x):
        if len(x) != len(self.levels):
            raise ValueError("The feature count must match the number of SJRH levels")
        pairs = [level.forward_with_diagnostics(feature) for level, feature in zip(self.levels, x)]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]
