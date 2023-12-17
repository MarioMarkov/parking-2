from utils.image_utils import extract_bndbox_values
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import datasets, models, transforms
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import os
from utils.image_utils import mAlexNet
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import os

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    
print(f"Using {device}")

model = "alex" #m_alex
image_dir= "./inference/images"
annotation_dir= "./inference/Annotations"
model_dir = "./models/final_alex_net_cnr.pth"
predicted_dir = "./predicted_images"

if not os.path.isdir(predicted_dir):
    os.mkdir(predicted_dir)


# Transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


if model == "m_alex":
    model = mAlexNet()
elif model  == "alex":
    model = models.alexnet(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=9216, out_features=256, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=256, out_features=128, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128, out_features=1, bias=True),
    )
else:
    raise Exception("Not a valid model type")


model = model.to(device)

model.load_state_dict(
    torch.load(model_dir, map_location=torch.device(device))
)
model.eval()

# Load coco
coco = COCO("./inference_coco/result.json")
output_dir = 'output_coco'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 
    
images_dir = "./inference_coco"    
image_paths = [os.path.join(images_dir, img['file_name']) for img in coco.loadImgs(coco.getImgIds())]

for image_path in image_paths:
   coco.getImgIds()



# start_time = time.time()
# for ann_id in coco.getAnnIds():
#     ann = coco.loadAnns(ann_id)[0]
#     image_id = ann['image_id']
#     image_info = coco.loadImgs(image_id)[0]
#     image_path = os.path.join('./inference_coco', image_info['file_name'])
#     #image_to_draw = cv2.imread(image_path)
   
    
#     # Load the image
#     img = Image.open(image_path)
    
#     img_to_draw = Image.open(image_path)
#     draw = ImageDraw.Draw(img)
    
#     mask = Image.new('1', (img.width, img.height), 0)
#     ImageDraw.Draw(mask).polygon(ann['segmentation'][0], outline=1, fill=1)
    
    
#     # Apply the mask to the image
#     result = Image.new('RGB', (img.width, img.height))
#     result.paste(img, mask=mask)
    
#     bbox = mask.getbbox()
    
    
#     # Crop the image to the bounding box
#     cropped_patch = result.crop(bbox)
#     img = transform(cropped_patch)
#     img = img.unsqueeze(0)
#     img = img.to(device)
#     with torch.no_grad():            
#         outputs = model(img)
#         #print(outputs)
#         #proba_max, preds = torch.max(outputs, 1)
#         preds = (torch.sigmoid(outputs) > 0.5).float()

#         is_busy = preds[0]
#         print(is_busy)
        
#         draw.polygon(ann['segmentation'][0], outline="red" if is_busy ==1 else "green", width=2)
#         annotated_filename = f"{ann_id}_annotated.png"
#         annotated_path = os.path.join(output_dir, annotated_filename)
#         img_to_draw.save(annotated_path)



