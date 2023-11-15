import torch
import time
import os
from tempfile import TemporaryDirectory
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn


def test_model(
    model,
    dataloaders,
    dataset_sizes,
    device,
    model_name,
):
    return -1
    
    