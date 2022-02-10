"""
Download a cat and dog image dataset and train a CNN to classify cats and dogs using Pytorch
"""

import os
import urllib.request
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import matplotlib.pyplot as plt

# Data download
DATASET_URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
DATA_DIR = os.path.join(os.getcwd(), 'data')
DOG_SYNSET_PREFIX = 'n02084071'
CAT_SYNSET_PREFIX = 'n02121808'

# Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5

if __name__ == '__main__':
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download and extract dataset
    dataset_path = os.path.join(DATA_DIR, 'kagglecatsanddogs.zip')
    if not os.path.exists(dataset_path):
        print('Downloading dataset...')
        urllib.request.urlretrieve(DATASET_URL, dataset_path)
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print('Done!')

    # Prepare dataset
    print('Preparing dataset...')

    # Create data transform
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ImageFolder(os.path.join(DATA_DIR, 'PetImages'), transform=data_transform)
    valid_dataset = ImageFolder(os.path.join(DATA_DIR, 'PetImages'), transform=data_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('Done!')

    # Create base model
    model = models.resnet18(pretrained=True)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Create new classifier
    classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
        nn.LogSoftmax(dim=1)
    )

    # Replace base classifier
    model.fc = classifier

    # Create loss function
    criterion = nn.NLLLoss()

    # Create optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

    # Move model to GPU
    model.to('cuda')

    # Train model
    print('Training model...')
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader):
            # Move images, labels to GPU
            images, labels = images.to('cuda'), labels.to('cuda')

            # Clear gradients
            optimizer.zero_grad()

            # Get model output
            output = model.forward(images)

            # Calculate loss
            loss = criterion(output, labels)

            # Propagate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Validation
        model.eval()
        validation_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                # Move images, labels to GPU
                images, labels = images.to('cuda'), labels.to('cuda')

                # Get model output
                output = model.forward(images)

                # Calculate loss
                v_loss = criterion(output, labels)
                validation_loss += v_loss.item()

                # Get probabilities
                probabilities = torch.exp(output)

                # Get top predicted class
                top_p, top_class = probabilities.topk(1, dim=1)

                # Compute accuracy
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}.. '
              f'Training loss: {running_loss / len(train_loader):.3f}.. '
              f'Validation loss: {validation_loss / len(valid_loader):.3f}.. '
              f'Accuracy: {accuracy / len(valid_loader):.3f}')

        # Reset running loss
        running_loss = 0

    print('Done!')

    # Save model
    model_path = os.path.join(DATA_DIR, 'dogs_vs_cats.pt')
    if not os.path.exists(model_path):
        print('Saving model...')
        torch.save(model, model_path)
        print('Done!')

    # Create test dataset
    test_dataset = ImageFolder(os.path.join(DATA_DIR, 'PetImages'), transform=data_transform)

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Test model
    print('Testing model...')
    with torch.no_grad():
        model.eval()
        test_loss = 0
        accuracy = 0
        for images, labels in test_loader:
            # Move images, labels to GPU
            images, labels = images.to('cuda'), labels.to('cuda')

            # Get model output
            output = model.forward(images)

            # Calculate loss
            t_loss = criterion(output, labels)
            test_loss += t_loss.item()

            # Get probabilities
            probabilities = torch.exp(output)

            # Get top predicted class
            top_p, top_class = probabilities.topk(1, dim=1)

            # Compute accuracy
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Testing loss: {test_loss / len(test_loader):.3f}.. '
          f'Accuracy: {accuracy / len(test_loader):.3f}')
