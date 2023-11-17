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
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        
        
    accuracy = running_corrects.to(torch.float32) / dataset_sizes["val"]
    accuracy
    
    return accuracy
    
    