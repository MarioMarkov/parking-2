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

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    
print(f"Using {device}")

model = "m_alex" #m_alex
image_dir= "./inference/images"
annotation_dir= "./inference/Annotations"
model_dir = "./models/m_alex_net_pk_best_acc.pth"
#predicted_dir = "./predicted_images"


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



## Inference video 

cap = cv2.VideoCapture("parking_video.mp4")
ret = True
frame_num = 0
step = 60
bndbox_values = extract_bndbox_values("./anotations_video.xml")
spots = {}
def draw_boxes(frame, bndbox_values, spots):
    for key in bndbox_values:
        xmin, ymin, xmax, ymax = bndbox_values[key]["xmin"], bndbox_values[key]["ymin"], bndbox_values[key]["xmax"], bndbox_values[key]["ymax"]
        if spots[key] == 1:
            # Busy - red
            frame = cv2.rectangle(
                            frame,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 0, 255),
                            2,
                        )
        else:
            frame = cv2.rectangle(
                            frame,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 255, 0),
                            2,
                        )
            
def predict_patch(values):
    xmin = int(values["xmin"])
    ymin = int(values["ymin"])
    xmax = int(values["xmax"])
    ymax = int(values["ymax"])
    patch = frame[ymin : ymax, xmin : xmax]
    image_pil = Image.fromarray(np.uint8(patch))

    img = transform(image_pil)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():                
        outputs = model(img)
        preds = (torch.sigmoid(outputs) > 0.6).float()
    return preds[0]
    

    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Every step frame refresh values
    if frame_num % step == 0:
        for key in bndbox_values:
            pred = predict_patch(bndbox_values[key])
            spots[key] = pred.item()

    # this frame is the image, for loop over all the spots    
    draw_boxes(frame, bndbox_values, spots)
    
    frame_num += 1
    cv2.imshow('Parking Lot', frame)
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()