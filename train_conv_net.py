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


# python train_alex.py --model_name=malex_net_pk_lot --data_dir=pk_lot_data --epochs=3 
FLAGS = argparse.ArgumentParser(description='Train Convnet')

FLAGS.add_argument('--data_dir', default="parking_data/", 
                   help='Location to the data path')
FLAGS.add_argument('--model_name', default="alex_net", 
                   help='Model name')
FLAGS.add_argument('--epochs', default=3, 
                   help='Epochs for model')
args = FLAGS.parse_args()

def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print(device)
    data_dir = args.data_dir

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
                transforms.CenterCrop(227),
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
    print(dataset_sizes)
   
    
    # Loading model weights
    model_ft = models.convnext_tiny(weights="IMAGENET1K_V1")
    
    model_ft.classifier = nn.Sequential(
            nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=768, out_features=2, bias=True)
    )
    
    
    # Getting the number of features in the last layer
    # num_ftrs = model_ft.classifier[-1].in_features
    
    # # # Making the output neurons to be 2 for 1 and 0
    # model_ft.classifier[-1] = nn.Linear(num_ftrs, 2)
    
    pytorch_train_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)

    print(f"Trainable params {pytorch_train_params}")
    
    print(model_ft)
    
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # model_ft = train_model(
    #     model_ft,
    #     criterion,
    #     optimizer_ft,
    #     exp_lr_scheduler,
    #     dataloaders,
    #     dataset_sizes,
    #     device,
    #     args.model_name,
    #     num_epochs= int(args.epochs),
    # )


if __name__ == "__main__":
    main()
