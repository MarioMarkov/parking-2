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

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

image_folder = "inference/parking_mag/"
annotation_folder = "inference/annotations/"
model_dir = "models/full_m_alex_net_pk_lot.pth"
predicted_images = "/predicted_images/"

# Annotations
xml_file = "inference/annotations/20230608_110054.xml"

if not os.path.isdir("predicted_images"):
    os.mkdir("predicted_images")


# Get Bounding box values
bndbox_values = extract_bndbox_values(xml_file)
# Transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# model_ft = models.alexnet(weights="IMAGENET1K_V1")
model = torch.load(model_dir, map_location=device)
model.eval()

model.to(device)
start_time = time.time()


# for image in image_folder
# find annotation with the same name in annotation folder
# run the whole anlaysis and generate image_with_boxes in folder
for image_filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_filename)

    image_to_draw = cv2.imread(image_path)
    full_image = Image.open(image_path)

    if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
        annotation_filename = os.path.join(
            annotation_folder,
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

                # img = Image.open(patch)
                # image_to_show = np.transpose(np.array(patch),(1, 2, 0))
                # plt.imshow(patch)
                # plt.show()

                img = transform(patch)
                img = img.unsqueeze(0)
                img = img.to(device)
                with torch.no_grad():
                    outputs = model(img)
                    _, preds = torch.max(outputs, 1)
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

                else:
                    # Free 0 green
                    cv2.rectangle(
                        image_to_draw,
                        (int(xmin), int(ymin)),
                        (int(xmax), int(ymax)),
                        (0, 255, 0),
                        2,
                    )
            print(image_filename)
            
            cv2.imwrite(f"predicted_images/{image_filename}", image_to_draw)


print("Execution time: %s seconds" % (time.time() - start_time))
