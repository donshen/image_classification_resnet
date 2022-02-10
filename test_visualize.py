import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

DATA_DIR = os.path.join(os.getcwd(), 'data')
model_path = os.path.join(DATA_DIR, 'dogs_vs_cats.pt')
model = torch.load(model_path)
model.eval()

# Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5

# Create data transform
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create test dataset
test_dataset = ImageFolder(os.path.join(DATA_DIR, 'PetImages'), transform=data_transform)

# Create test dataloader
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Visualize predictions
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
# axes = axes.flatten()
with torch.no_grad():
    model.eval()
    for i, (img, label) in enumerate(test_loader):
        if i >= 5:
            break
            
        # Move images, labels to GPU
        img, label = img.to('cuda'), label.to('cuda')

        # Get model output
        output = model.forward(img)

        # Get probabilities
        probabilities = torch.exp(output)

        # Get top predicted class
        top_p, top_class = probabilities.topk(1, dim=1)

        img, label, top_class = img.cpu().numpy()[0], label.cpu().numpy()[0], top_class.cpu().numpy()[0][0]
        
        axes[i, 0].imshow(img[2, :, :])
        axes[i, 0].set_title(f'Label: {int(label)}')
        axes[i, 1].bar(range(2), [1 - top_class, top_class])
        axes[i, 1].set_xticks([0, 1])
        axes[i, 1].set_xticklabels(['Cat', 'Dog'])
        axes[i, 1].set_title('Prediction')
        
plt.show()