import numpy as np
import torch
import albumentations as A
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from . import config
from typing import List
from torchvision import transforms
from torchvision.transforms import ToPILImage
from copy import deepcopy

torch_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def resize_img(img):
    """
    img:  Pillow image
    """
    h, w = img.height, img.width
    if h >= w:
        ratio = config.input_height / h
        new_h, new_w = int(h * ratio), int(w * ratio)        
    else:
        ratio = config.input_width / w
        new_h, new_w = int(h * ratio), int(w * ratio)

    img = img.resize((new_w, new_h), Image.BILINEAR)
    return img, (w, h)


def pad_img(img):
    h = img.height
    w = img.width
    img = np.array(img)
    img = np.pad(img, pad_width=((0, config.input_height - h), (0, config.input_width - w), (0, 0)), mode="constant")
    img = Image.fromarray(img)
    assert img.height == config.input_height
    assert img.width == config.input_width
    return img


def resize_and_padding(img, return_window=False):
    img, (ori_w, ori_h) = resize_img(img)
    w = img.width
    h = img.height
    padding_window = (w, h)
    img = pad_img(img)

    if not return_window:
        return img
    else:
        return img, padding_window, (ori_w, ori_h)



albumentation_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, rotate_limit=10, p=0.7),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.ColorJitter(p=0.5)
    ], p=0.8),

    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1,
    bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)

albumentation_minimal_transform = A.Compose([
    A.LongestMaxSize(max_size=config.input_height, interpolation=1, p=1),
    A.PadIfNeeded(
        min_height=config.input_height,
        min_width=config.input_height,
        border_mode=0,
        value=(0, 0, 0),
        position="top_left"
    ),
],
    p=1, bbox_params=A.BboxParams(format="pascal_voc", min_area=0.1)
)


def data_augmentation(img, bboxes=None, return_window=False):
    if isinstance(img, Image.Image):
        img = np.array(img)
    ori_h, ori_w = img.shape[0:2]
    if bboxes is not None:
        transformed = albumentation_transform(image=img, bboxes=bboxes)
        img = transformed["image"]
        bboxes = transformed["bboxes"]
    else:
        transformed = albumentation_transform(image=img, bboxes=[[1, 2, 3, 4, 0]])
        img = transformed["image"]

    max_side = max(ori_h, ori_w)
    ratio = config.input_width / max_side
    # to be used by pillow, so represented in (0,0, w,h)
    padding_window = (0, 0, ori_w * ratio, ori_h * ratio)

    if not return_window:
        if bboxes is not None:
            return img, bboxes
        else:
            return img
    else:
        return img, bboxes, padding_window, (ori_w, ori_h)


def data_minimal_augmentation(img, boxes=None):
    # in order to fit into the network
    if isinstance(img, Image.Image):
        img = np.array(img)
    if boxes is None:
        boxes = [[1, 2, 3, 4, 0]]
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"]
    else: 
        transformed = albumentation_minimal_transform(image=img, bboxes=boxes)
        return transformed["image"], transformed["bboxes"]

class AnnotationDataset(Dataset):
    def __init__(self, mode="train"):
        assert mode in ["train", "test"]
        self.mode = mode
        super(AnnotationDataset, self).__init__()
        self.annotations = {}
        self.cls_names = config.labels

        if self.mode == "train":
            annotation_files = config.train_annotation_files
        else:
            annotation_files = config.test_annotation_files

        for annotation_file in annotation_files:
            with open(annotation_file) as f:
                for line in f:
                    line = line.replace("\n", "").split(",")
                    img_path, x1, y1, x2, y2, cls_name = line
                    if x1 + y1 + x2 + y2 + cls_name == "":
                        if self.annotations.get(img_path, None) is None:
                            self.annotations.update({img_path: []})
                    else:
                        xmin = float(x1)
                        ymin = float(y1)
                        xmax = float(x2)
                        ymax = float(y2)
                        cls_index = self.cls_names.index(cls_name) + 1
                        box = [xmin, ymin, xmax, ymax, cls_index]

                        if self.annotations.get(img_path, None) is None:
                            self.annotations.update({img_path: [box]})
                        else:
                            self.annotations[img_path].append(box)

        # [imagesbasename, [[bbox, cls_index]]]
        self.data: List[str, List[List[float]]] = [[k, v] for k, v in self.annotations.items()]

    def __getitem__(self, index):
        dataset = self.data
        data = dataset[index]
        img_path, boxes = data

        img = Image.open(img_path)

        # draw1 = ImageDraw.Draw(img)
        # for box in boxes:
        #     xmin, ymin, xmax, ymax, _ = box
        #     draw1.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 255, 0, 150), width=3)
        # img.show()

        if self.mode == "train":
            img = Image.open(img_path)
            if len(boxes) >0:
                img, new_boxes = data_augmentation(img, boxes)

                # img = Image.fromarray(img)
                # draw2 = ImageDraw.Draw(img)  # img: PIL.Image.Image
                # for box in new_boxes:
                #     xmin, ymin, xmax, ymax, _ = box
                #     draw2.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 255, 0, 150), width=3)
                # img.show()

                img = torch_img_transform(img)
                if len(new_boxes) == 0:
                    return img, [], []
                new_boxes = np.array(new_boxes)

                boxes_ = torch.as_tensor(new_boxes[..., 0:4]).float()
                targets = torch.as_tensor(new_boxes[..., 4]).float()

                return img, boxes_, targets
            else:
                img = data_augmentation(img)
                img = torch_img_transform(img)
                return img, torch.as_tensor([]), torch.as_tensor([])
        else:
            if len(boxes) > 0:
                img, boxes = data_minimal_augmentation(img, boxes)
                img = torch_img_transform(img)
                boxes = np.array(boxes)

                return img, boxes, img_path
            else:
                img = data_minimal_augmentation(img)
                img = torch_img_transform(img)
                return img, torch.as_tensor([]), img_path

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = AnnotationDataset()
    a, b = dataset[0]
    print(a, b)
