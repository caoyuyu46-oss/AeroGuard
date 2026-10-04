# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):
    

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "segment"
        self.args.plots = False  
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
