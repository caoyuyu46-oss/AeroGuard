# Ultralytics YOLO 🚀, AGPL-3.0 license


from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    

    def __init__(self, model="rtdetr-l.pt") -> None:
        
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            }
        }
