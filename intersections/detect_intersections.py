import xml.etree.ElementTree as ET
import cv2


def extract_bndbox_values_yolo(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    bndbox_values = {}

    for i, obj in enumerate(root.findall("coordinates")):
        xmin = float(obj.find("xmin").text)
        ymin = float(obj.find("ymin").text)
        xmax = float(obj.find("xmax").text)
        ymax = float(obj.find("ymax").text)
        bndbox_values[str(i)] = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

    return bndbox_values


yolo_detections = extract_bndbox_values_yolo("intersections/detections.xml")
img = cv2.imread("intersections/pg_survailance_alex.jpg")

for d in yolo_detections:
    coordinates = yolo_detections[d]
    cv2.rectangle(
        img,
        (int(coordinates["xmin"]), int(coordinates["ymin"])),
        (int(coordinates["xmax"]), int(coordinates["ymax"])),
        (255, 0, 0),
        2,
    )

cv2.imwrite("yolo.jpg", img)
