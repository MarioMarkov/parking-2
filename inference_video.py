import cv2
import numpy as np
import sys
import os
from utils.video_utils import get_parking_spots_bboxes
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Using {device}")
   

# Transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



video_path = "video/parking_1920_1080.mp4"
mask = "video/mask_1920_1080.png"
model_ft = torch.load("models/full_tiny_alex_net_pk_lot.pth", map_location=device)

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

conneced_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(conneced_components)


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


spots_status = [None for j in spots]
previous_frame = None

ret = True
step = 120
diffs = [None for j in spots]

frame_num = 0
while ret:
    ret, frame = cap.read()

    # compare to previous frame
    if frame_num % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w]

            diffs[spot_index] = calc_diff(
                spot_crop, previous_frame[y1 : y1 + h, x1 : x1 + w]
            )

    if frame_num % step == 0:
        _arr = range(len(spots))
        for spot_index in _arr:
            spot = spots[spot_index]
            x1, y1, w, h = spot

            spot_crop = frame[y1 : y1 + h, x1 : x1 + w]

            image_pil = Image.fromarray(np.uint8(spot_crop))
            # plt.imshow(image_pil)
            # plt.show()
            # spot_crop = frame.crop((x1, y1, x1 + w, y1 + h))

            img = transform(image_pil)
            img = img.unsqueeze(0)
            img = img.to(device)
            with torch.no_grad():
                outputs = model_ft(img)
                _, preds = torch.max(outputs, 1)
                is_busy = preds[0]

            spots_status[spot_index] = is_busy

    if frame_num % step == 0:
        previous_frame = frame.copy()

    # Drawing the rectangles
    for spot_index, spot in enumerate(spots):
        is_busy = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]

        if is_busy:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    frame_num += 1

    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cap.destroyAllWindows()
