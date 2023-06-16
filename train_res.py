import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import argparse
from PIL import Image
from tempfile import TemporaryDirectory
from train import train_model
from utils.image_utils import imshow
from utils.viz_model  import visualize_model_predictions, visualize_model
plt.ion()

FLAGS = argparse.ArgumentParser(description='Train ResNet')

FLAGS.add_argument('--data_dir', default="parking_data/", 
                   help='Location to the data oath')
FLAGS.add_argument('--model_name', default="res_net", 
                   help='Name of the model')
args = FLAGS.parse_args()


def train(device, model, dataloaders):
        
    model.to(device)
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(
        model,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        device,
        args.model_name,
        num_epochs=5,
    )

if __name__ == "__main__":
    data_dir = "parking_data/"
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(device)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=16, shuffle=True, num_workers=0
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    
    model_ft = models.resnet18(weights="IMAGENET1K_V1")

    num_ftrs = model_ft.fc.in_features
    
    # Here the size of each output sample is set to 2.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    train(device, model_ft, dataloaders)

    model_ft.load_state_dict(
        torch.load("best_model_res_net.pth", map_location=torch.device(device))
    )
    model_ft.to(device)
    print("Model loaded!")
    print("Last layer: ", model_ft.fc)
    visualize_model(model_ft, dataloaders, device=device)
    # visualize_model_predictions(
    #     model_ft, img_path="R_2015-11-21_07.40_C01_184.jpg",
    #     data_transforms= data_transforms,
    #     device=device, 
    # )
    
