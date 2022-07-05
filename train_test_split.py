import os
import random
import json
import random
from glob import glob 
save_train_path = "annotations/train.txt"
save_test_path = "annotations/test.txt"
training_set_ratio = 0.85

with open("coco_annotation.json", "r") as f:
    coco  = json.load(f)
    

images = coco["images"]
categories = coco["categories"]
annotations = coco["annotations"]

imageId_imagepath = {}
catId_label = {}

for img_path in images:
    imageId_imagepath.update({img_path["id"]: img_path["path"]})
for cat in categories:
    catId_label.update({cat["id"]: cat["name"]})


anno_dict = {}





for annotation in annotations:
    image_id = annotation["image_id"]
    cat_id = annotation["category_id"]
    xywh = annotation["bbox"]
    xmin = xywh[0]
    ymin = xywh[1]
    xmax = xmin + xywh[2]
    ymax = ymin + xywh[3]
    img_path = imageId_imagepath[image_id]
    label = catId_label[cat_id]
    data = [xmin, ymin, xmax, ymax, label]
    
    if anno_dict.get(img_path, None) is None:
        anno_dict.update({img_path: [data]})
    else:
        anno_dict[img_path].append(data)

normal_imgs = glob("datasets/normal/*.jpg")

set_of_rusty_imgs = list(anno_dict.keys())
random.shuffle(set_of_rusty_imgs)
random.shuffle(normal_imgs)
n_train = int(0.85 * len(set_of_rusty_imgs))
train_imgs = set_of_rusty_imgs[:n_train]
test_imgs = set_of_rusty_imgs[n_train:]

for mode, imgs in [("train", train_imgs), ("test", test_imgs)]:
    lines = []
    with open(f"annotations/{mode}.txt", "w+") as f:
        for img_path in imgs:
            anno = anno_dict[img_path]
            for x1,y1,x2,y2,label in anno:
                lines.append(",".join([img_path, str(x1), str(y1), str(x2), str(y2), label]))
        for i in range(int(3*len(imgs))):
            lines.append("{},,,,,".format(normal_imgs[i]))
        lines = "\n".join(lines)
        f.write(lines)
            
    
    



    
    
    
