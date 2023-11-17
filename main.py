import os
import torch
import argparse
import torchvision
import torch.nn as nn
from test import test_model

from train import train_model
#from test import test_model

from torchvision import datasets, models, transforms

from utils.image_utils import mAlexNet


#train
# python main.py --model_name=malex_net_pk_lot --data_dir=pk_lot_data --epochs=2 --dataset=both -train 

#test 
# python main.py --model_name=malex_net_combined_state_dict --dataset=cnr_park
FLAGS = argparse.ArgumentParser(description="Train Model")

FLAGS.add_argument('-train',
                    action='store_true')

FLAGS.add_argument(
    "--pk_lot_dir", default="pk_lot_data/", help="Location to the pk_lot data path"
)
FLAGS.add_argument(
    "--cnr_park_dir", default="cnr_parking_data/", help="Location to the cnr_park data path"
)
FLAGS.add_argument(
    "--dataset", default="pk_lot", help="On which dataset to train or test. Possible values are: 'both', 'pk_lot', 'cnr_park'"
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

    print("Running on device: ", device) 

    pk_lot_dir = args.pk_lot_dir
    cnr_park_dir = args.cnr_park_dir

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

    image_datasets_pk_lot = {
        x: datasets.ImageFolder(os.path.join(pk_lot_dir,x), data_transforms[x])
        for x in ["train", "val"]
    }
    image_datasets_cnr_park = {
        x: datasets.ImageFolder(os.path.join(cnr_park_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    
    if args.dataset == "both":
        train_dataset = torch.utils.data.ConcatDataset([image_datasets_pk_lot["train"], image_datasets_cnr_park["train"]])
        val_dataset = torch.utils.data.ConcatDataset([image_datasets_pk_lot["val"], image_datasets_cnr_park["val"]])
    elif args.dataset  == "pk_lot":
        train_dataset = image_datasets_pk_lot["train"]
        val_dataset = image_datasets_pk_lot["val"]
    elif args.dataset  == "cnr_park":
        train_dataset = image_datasets_cnr_park["train"]
        val_dataset = image_datasets_cnr_park["val"]
    else:
        print("Not a valid image dataset provided. Possible values are: 'both', 'pk_lot', 'cnr_park'")

    dataloaders= {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
    }

    dataset_sizes = {"train": len(train_dataset) , "val": len(val_dataset)}
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

    
    if args.train:
        model = train_model(
            model,
            dataloaders,
            dataset_sizes,
            device,
            args.model_name,
            num_epochs=int(args.epochs),
        )
    else: 
        results = test_model( model,
            dataloaders,
            dataset_sizes,
            device,
            args.model_name)
        print(f"Model accuracy: {round(results.item(),5)*100}%")

if __name__ == "__main__":
    main()