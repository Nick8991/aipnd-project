from PIL import Image
from torch import optim, nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms, models



def process_image(batch_sze):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(60),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_transforms_valid = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms_train)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)

    image_datatset = [train_image_datasets, valid_image_datasets]

    # Using the image datasets and the transforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_sze, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=batch_sze, shuffle=True)

    image_loader = [train_loader, valid_loader]


    return image_datatset, image_loader

def modelDefinition(model_name, num_hidden_layers):
    if model_name == 'vgg':
        model = models.vgg16(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        if num_hidden_layers == 2:
            classifier = nn.Sequential(nn.Linear(25088,1568),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(1568,392),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(392,102),
                           nn.LogSoftmax(dim=1))
            
        elif num_hidden_layers == 3:
            classifier = nn.Sequential(nn.Linear(25088,3136),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(3136,1568),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(1568,392),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.2),
                                    nn.Linear(392,102),
                                    nn.LogSoftmax(dim=1))
            
    
    elif model_name == 'AlexNet':
        model = models.alexnet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        if num_hidden_layers == 1:
            classifier = nn.Sequential(
                nn.Linear(9216, 4096),  # AlexNet has a different input size
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(4096, 102),
                nn.LogSoftmax(dim=1)
            )
        elif num_hidden_layers == 2:
            classifier = nn.Sequential(
                nn.Linear(9216, 4096),  # Input layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(4096, 1024),  # First hidden layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 102),   # Second hidden layer
                nn.LogSoftmax(dim=1)    # Output layer
            )
        elif num_hidden_layers == 3:
            classifier = nn.Sequential(
                nn.Linear(9216, 4096),  # Input layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(4096, 2048),  # First hidden layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(2048, 1024),  # Second hidden layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 512),   # Third hidden layer
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, 102),    # Output layer
                nn.LogSoftmax(dim=1)
            )
        model.classifier = classifier

return model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)








