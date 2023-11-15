import os
import torch
import argparse
import torchvision
import torch.nn as nn

from train import train_model
#from test import test_model

from torchvision import datasets, models, transforms

from utils.image_utils import mAlexNet

train = True
# python main.py --model_name=malex_net_pk_lot --data_dir=pk_lot_data --epochs=2
FLAGS = argparse.ArgumentParser(description="Train Model")

FLAGS.add_argument(
    "--data_dir", default="pk_lot_data/", help="Location to the data path"
)
FLAGS.add_argument("--model_name", default="alex_net", help="Model name")
FLAGS.add_argument("--epochs", default=3, help="Epochs")
FLAGS.add_argument("--batch_size", default=16, help="Batch size")
args = FLAGS.parse_args()


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    print("Training on device: ", device)

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
            image_datasets[x],
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=0,
        )
        for x in ["train", "val"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    print("Size of train and test: ", dataset_sizes)

    if args.model_name == "mobile_net":
        model = torchvision.models.quantization.mobilenet_v2(
            weights= "DEFAULT", quantize=True
        )
        model.classifier[-1].out_features = 2

    else:
        # model = models.alexnet(weights="IMAGENET1K_V1")
        # model.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=9216, out_features=256, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Linear(in_features=256, out_features=128, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=128, out_features=2, bias=True),
        # )
        model = mAlexNet()

    # Send model to device
    model = model.to(device)

    print("Model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad) )

    
    if train == True:
        model = train_model(
            model,
            dataloaders,
            dataset_sizes,
            device,
            args.model_name,
            num_epochs=int(args.epochs),
        )
    else: 
        #test_model()
        pass

if __name__ == "__main__":
    main()