# Ultralytics YOLO 🚀, AGPL-3.0 license


from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.utils.torch_utils import model_info

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):
    

    def __init__(self, model="yolo_nas_s.pt") -> None:
        
        assert Path(model).suffix not in {".yaml", ".yml"}, "YOLO-NAS models only support pre-trained models."
        super().__init__(model, task="detect")

    def _load(self, weights: str, task=None) -> None:
        
        import super_gradients

        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model = torch.load(attempt_download_asset(weights))

        elif suffix == "":
            self.model = super_gradients.training.models.get(weights, pretrained_weights="coco")

        
        def new_forward(x, *args, **kwargs):
            
            return self.model._original_forward(x)

        self.model._original_forward = self.model.forward
        self.model.forward = new_forward

        
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  
        self.model.yaml = {}  
        self.model.pt_path = weights  
        self.model.task = "detect"  

    def info(self, detailed=False, verbose=True):
        
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        
        return {"detect": {"predictor": NASPredictor, "validator": NASValidator}}
