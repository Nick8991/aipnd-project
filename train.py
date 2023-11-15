from torch import optim, nn
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import model_classifier
import json



def process_image(batch_sze):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    # test_dir = data_dir + '/test'

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

def modelDefinition(model_name,num_hidden_layers, learning_rate):
    network = ''
    if model_name == 'vgg':
        model = models.vgg16(pretrained=True)
        network = 'vgg'

        for param in model.parameters():
            param.requires_grad = False

        if num_hidden_layers == 2:
            classifier = model_classifier.vgg1()
            hidden_layers = [1568,392]
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            return model, hidden_layers, network,optimizer

            
        elif num_hidden_layers == 3:
            classifier = model_classifier.vgg2()
            hidden_layers = [3136,1568,392]
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            return model, hidden_layers, network,optimizer
            
    
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)

        network = 'alexnet'
        for param in model.parameters():
            param.requires_grad = False

        if num_hidden_layers == 1:
            classifier = model_classifier.alexnet1()
            hidden_layers = [4096]
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            return model, hidden_layers, network,optimizer
        
        elif num_hidden_layers == 2:
            classifier = model_classifier.alexnet2()
            hidden_layers = [4096,1024]
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            return model, hidden_layers, network,optimizer
        
        elif num_hidden_layers == 4:
            classifier = model_classifier.alexnet3()
            hidden_layers = [4096,2048,1024,512]
            model.classifier = classifier
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            return model, hidden_layers, network,optimizer


def training_model(model, numberOfEpochs, image_loader,optimizer, deviceToUse):
    device = torch.device(deviceToUse)
    criterion = nn.NLLLoss()
    model = model.to(device)
    epochs = numberOfEpochs
    steps = 0
    running_loss = 0
    print_every = 5


    for e in range(numberOfEpochs):
        for images, labels in image_loader[0]:
            steps += 1
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model.forward(images)

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()

                valid_loss = 0
                accuracy = 0

                for images, labels in image_loader[1]:
                    images, labels = images.to(device), labels.to(device)

                    logits = model.forward(images)
                    batch_loss = criterion(logits, labels)

                    valid_loss += batch_loss.item()


                    #calculating accuracy
                    ps = torch.exp(logits)
                    top_p, top_classes = ps.topk(1, dim=1)
                    equals = top_classes == labels.view(*top_classes.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}..."
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(image_loader[1]):.3f}.. "
                    f"Validation accuracy: {accuracy/len(image_loader[1]):.3f}")
                running_loss = 0
                model.train()

        
    return model

def save_model(model, image_datatset, network, num_hidden_layer):
    model.class_to_idx = image_datatset[0].class_to_idx
    if network == 'vgg':
        # Save the checkpoint
        checkpoint = {'input_size': 25088,
                    'output_size': 102,
                    'hidden_layers': num_hidden_layer,
                    'state_dict': model.state_dict(),
                    'class_to_idx': image_datatset[0].class_to_idx}
    elif network == 'alexnet':
        # Save the checkpoint
        checkpoint = {'input_size': 9216,
                    'output_size': 102,
                    'hidden_layers': num_hidden_layer,
                    'state_dict': model.state_dict(),
                    'class_to_idx': image_datatset[0].class_to_idx}
    data_dict = {'file_name':f'{network}_model.pth', 'hidden_layers':f'{num_hidden_layer}', 'network':network}
    torch.save(checkpoint, f'{network}_model.pth')

    with open('load_point.json', 'w') as f:
        json.dump(data_dict, f)


if __name__=="__main__":
    print("______________________________________________________________________Program Launched_________________________________________________________________________________\n")
    network_choice = None
    while network_choice != 'vgg' and  network_choice != 'alexnet':
        network_choice = input("Select a convolutional Neural Network model to use,  [vgg, alexnet]\n").lower()
    batch_size = 0
    batch_list = [16,32,64]
    while batch_size not in batch_list:
        batch_size = int(input(f"Select a batch size from {batch_list}\n"))

    image_datatset_list, image_loader_list =  process_image(batch_size)

    num_hidden_layers = 0
    list1 = [2,3]
    list2 = [1,2,4]
    if network_choice == 'alexnet':
        while num_hidden_layers not in list2:
            num_hidden_layers = int(input("\n Select number of hidden layers from list [1,2, 4]\n"))

    elif network_choice == 'vgg':
        while num_hidden_layers not in list1:
            num_hidden_layers = int(input("\n Select number of hidden layers from list [2,3]\n"))
    learn_rate = 0
    while not (0.001 <= learn_rate <= 0.01):
        learn_rate = float(input("Enter a number between 0.001 and 0.01: "))
        
    model, hidden_layers, network,optimizer = modelDefinition(network_choice,num_hidden_layers,learn_rate)

    numberOfEpochs = 0
    while numberOfEpochs not in range(1,21):
        numberOfEpochs = int(input("\n Enter an epoch number between 1 and 20\n"))
    deviceToUse = None
    list_devices = ['cpu', 'gpu']
    while deviceToUse not in list_devices:
        deviceToUse = input(f"\n Would you like to run the script on the 'gpu' or 'cpu'\n").lower()
    if torch.cuda.is_available() and deviceToUse == 'gpu':
        deviceToUse = 'cuda'
        model = training_model(model, numberOfEpochs, image_loader_list,optimizer, deviceToUse)
    elif deviceToUse == 'cpu':
        model = training_model(model, numberOfEpochs, image_loader_list,optimizer, deviceToUse)

    save_model(model, image_datatset_list, network, hidden_layers)

    print("\n____________________________________model saved_________________________________________________\n")
    print("\n\n\n____________________________________End of Script_________________________________________________\n")