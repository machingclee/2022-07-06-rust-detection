import torch
from src.train import train
from src.faster_rcnn import FasterRCNN
from src.device import device
from src.visualize import visualize
from src.visualize import inferenceg
from PIL import Image

def main():
    model_path = None

    faster_rcnn = FasterRCNN().to(device)

    if model_path is not None:
        faster_rcnn.load_state_dict(torch.load(model_path))
    faster_rcnn.train()

    train(
        faster_rcnn,
        lr=1e-5,
        start_epoch=1,
        epoches=50,
        unfreeze_after_epoch=None
    )

    # faster_rcnn.eval()
    # visualize(faster_rcnn)
        
def inference_on_image():
    model_path = "pths/model_epoch_50.pth"
    faster_rcnn = FasterRCNN().to(device)
    faster_rcnn.load_state_dict(torch.load(model_path))  
    faster_rcnn.eval()
    img_path_nonrust = "score_test_pool/normal/signboard_id150.jpg"
    img_path_rust = "score_test_pool/rust/rust_id83.jpg"
    
    score1 = inference(faster_rcnn, Image.open(img_path_rust))
    score2 = inference(faster_rcnn, Image.open(img_path_nonrust))
    
    
    print("normal", score2)
    print("rust", score1)

if __name__ == "__main__":
    inference_on_image()