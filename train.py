import torch
import time
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np

backend = "qnnpack"

def train_model(
    model,
    dataloaders,
    dataset_sizes,
    device,
    model_name,
    num_epochs=3,
):
    since = time.time()
    #torch.backends.quantized.engine = backend

    #Use BCEWithLogitsLoss() instead of BCELoss() since BCEWithLogitsLoss()already includes a sigmoid layer
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()

    # Define optimizer for nn
    optimizer = optim.AdamW(model.parameters(),lr=0.001, fused=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #best_model_params_path = os.path.join("best_model_statedict.pt")
    working_dir = os.getcwd()
    print(f"Model save path (working dir): {working_dir}")
    
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)
        
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)#.unsqueeze(1).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs).squeeze(dim=1)
                    #_, preds = torch.max(outputs, 1)
                    #preds = np.where(outputs > 0, 1, 0)
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.to(torch.float32) / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}")

            # If the current accuracy is better than the last best acc 
            # deep copy the model
            if phase == "val" and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                torch.save(
                    model.state_dict(), f"{working_dir}/{model_name}_state_dict.pth"
                )
                torch.save(model, f"{working_dir}/full_{model_name}.pth")

        print()
        torch.save(model.state_dict(), f"{working_dir}/final_{model_name}.pth")

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    print(f"Best val accuracy: {best_accuracy:4f}")
    # load best model weights
    model.load_state_dict(torch.load( f"{working_dir}/final_{model_name}.pth"))
    return model