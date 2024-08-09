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
image_dir= "./inference-label-studio/images"
annotation_dir= "./inference-label-studio/Annotations"
model_path = "./models/m_alex_net_both_best_acc.pth"
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
    torch.load(model_path, map_location=torch.device(device))
)
model.eval()


start_time = time.time()
# for image in image_folder
# find annotation with the same name in annotation folder
# run the whole anlaysis and generate image_with_boxes in folder
for image_filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_filename)

    image_to_draw = cv2.imread(image_path)
    #print("Path: ", image_path)
    full_image = Image.open(image_path)
    full_image = full_image.convert('RGB')
    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        annotation_filename = os.path.join(
            annotation_dir,
            image_filename.replace(".jpg", ".xml").replace(".png", ".xml"),
        )

        # Check if the annotation file exists
        if os.path.isfile(annotation_filename):
            bndbox_values = extract_bndbox_values(annotation_filename)

            for key in bndbox_values:
                values = bndbox_values[key]
                # Extract coordinates from the bounding box
                xmin = int(values["xmin"])
                ymin = int(values["ymin"])
                xmax = int(values["xmax"])
                ymax = int(values["ymax"])
                # Crop patch for the image
                patch = full_image.crop((xmin, ymin, xmax, ymax))
                #print(type(patch))
                # img = Image.open(patch)
                # image_to_show = np.transpose(np.array(patch),(1, 2, 0))
                # plt.imshow(patch)
                # plt.show()
                
                img = transform(patch)
                img = img.unsqueeze(0)
                img = img.to(device)
                with torch.no_grad():
                    
                    outputs = model(img)
                    #print(outputs)
                    #proba_max, preds = torch.max(outputs, 1)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    is_busy = preds[0]

                    if is_busy == 1:
                        # Busy 1 Red
                        cv2.rectangle(
                            image_to_draw,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 0, 255),
                            2,
                        )
                        # Draw black background rectangle
                        # cv2.rectangle(image_to_draw, (xmin, ymin), (xmin, ymin+10), (0,0,0), -1)
                        # cv2.putText(image_to_draw, str(round(torch.sigmoid(outputs).item(),2)), (xmin, ymin-10), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

                    else:
                        # Free 0 green
                        cv2.rectangle(
                            image_to_draw,
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 255, 0),
                            2,
                        )
                        #cv2.putText(image_to_draw, str(round(torch.sigmoid(outputs).item(),2)), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

            print(image_filename)

            cv2.imwrite(f"{predicted_dir}/{image_filename}", image_to_draw)


print("Execution time: %s seconds" % (time.time() - start_time))
