# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

import torch

from .head import Detect
from .aeroguard import SJRHCore
from ultralytics.utils.tal import make_anchors


class Detect_SJRH(Detect):
    def __init__(self, nc=80, hidden=96, max_offset_cells=1.5, shared_sampling=False, quality_scoring=False, quality_eta=0.5, ch=()):
        super().__init__(nc, ch)
        del self.cv2, self.cv3
        self.core = SJRHCore(nc, ch, self.reg_max, hidden, max_offset_cells, shared_sampling)
        self.no = self.reg_max * 4 + self.nc + 1
        self.quality_scoring = bool(quality_scoring)
        self.quality_eta = float(quality_eta)
        self.has_quality = True

    def forward(self, x):
        raw = [torch.cat(values, 1) for values in self.core(x)]
        if self.training:
            return raw
        y = self._inference(raw)
        return y if self.export else (y, raw)

    def _inference(self, x):
        shape = x[0].shape
        merged = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls, quality = merged.split((self.reg_max * 4, self.nc, 1), 1)
        decoded = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        scores = cls.sigmoid()
        if self.quality_scoring and self.quality_eta != 0:
            scores = scores * quality.sigmoid().pow(self.quality_eta)
        return torch.cat((decoded, scores), 1)

    def bias_init(self):
        return None

    def set_quality_scoring(self, enabled=True, eta=0.5):
        self.quality_scoring = bool(enabled)
        self.quality_eta = float(eta)
        return self

