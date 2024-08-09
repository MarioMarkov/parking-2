import torch
from tqdm import tqdm

def test_model(
    model,
    dataloaders,
    dataset_sizes,
    device,
    model_name,
):
    model.load_state_dict(torch.load(f"models\{model_name}.pth"))
    model.to(device)
    
    running_corrects = 0
    for inputs, labels in tqdm(dataloaders["val"]):
        
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            #_, preds = torch.max(outputs, 1)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            preds = preds.squeeze().int()  # Remove dimensions of size 1

            running_corrects += torch.sum(preds == labels).item()

    # print("Corrects", running_corrects)
    # print("dataset size", dataset_sizes["val"])

    accuracy = running_corrects / dataset_sizes["val"]
    accuracy
    
    return accuracy
    
    