import torch 
from PIL import Image
from utils.image_utils import imshow
import matplotlib.pyplot as plt 

def visualize_model_predictions(model, img_path, device, data_transforms):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms["val"](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2, 2, 1)
        ax.axis("off")
        ax.set_title(f"Predicted: {preds[0]}")
        imshow(img.cpu().data[0])

        model.train(mode=was_training)
        
def visualize_model(model, dataloaders, device, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {preds[j]}")
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        plt.show()
    plt.close()