from .src.faster_rcnn import FasterRCNN
from .src.device import device
from .src import config
from .src.visualize import DetectionResult, inference
import torch

class RustClassifier:
    faster_rcnn = None
    
    def get_faster_rcnn(self):
        if RustClassifier.faster_rcnn is not None:
            return RustClassifier.faster_rcnn
        
        faster_rcnn = FasterRCNN().to(device)
        model_path = config.serve_weight_path
        faster_rcnn.load_state_dict(torch.load(model_path))
        faster_rcnn.eval()
        RustClassifier.faster_rcnn = faster_rcnn
        return RustClassifier.faster_rcnn      

    def inference_on_image(self,image_path):
        # type: (str) -> DetectionResult
        faster_rcnn = self.get_faster_rcnn()
        result = inference(faster_rcnn, image_path)
        return result