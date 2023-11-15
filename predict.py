from PIL import Image
import numpy as np
import torch
from torchvision import models
import model_classifier
import json



def find_model_file():
    # Load the JSON file
    with open('load_point.json', 'r') as f:
        data = json.load(f)

    # Access the variables
    file_name = data['file_name']
    hidden_layers = str(data['hidden_layers'])
    network = data['network']

    return file_name, hidden_layers, network


def load_checkpoint(filepath,hidden_layers,network):
    print(hidden_layers, network)
    classifier = None
    checkpoint = torch.load(filepath)
    model = None

    
    
    if network == 'vgg':
        model = models.vgg16(pretrained=True)
        model.class_to_idx = checkpoint['class_to_idx']
        for param in model.parameters():
            param.requires_grad = False
        if hidden_layers == '[1568,392]':
            classifier = model_classifier.vgg1()
        elif hidden_layers == '[3136,1568,392]':
            classifier = model_classifier.vgg2()

    elif network == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.class_to_idx = checkpoint['class_to_idx']
        for param in model.parameters():
            param.requires_grad = False
        print('its an elexnet network')
        if hidden_layers == '[4096]':
            print(model_classifier.alexnet1())
            classifier = model_classifier.alexnet1()
        elif hidden_layers == '[4096,1024]':
            classifier = model_classifier.alexnet2()
        elif hidden_layers == '[4096,2048,1024,512]':
            classifier = model_classifier.alexnet3()
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    # Put the model in evaluation mode
    #model.eval()
    
    return model



def image_processing(image_path):
    # Load the image using PIL
    image = Image.open(image_path)
    
    # Resize the image, keeping the aspect ratio
    shortest_side = 256
    image.thumbnail((shortest_side, shortest_side))
    
    # Crop out the center 224x224 portion of the image
    left_margin = (image.width - 224) / 2
    top_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    bottom_margin = top_margin + 224
    image = image.crop((left_margin, top_margin, right_margin, bottom_margin))
    
    # Convert color channels to floats in the range [0, 1]
    np_image = np.array(image) / 255.0
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to have color channel as the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert numpy array to PyTorch tensor
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    return tensor_image


def load_cat_names():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def predict(processed_image, model,device_type, topk=5):
    # Use the model for prediction
    model.eval()
    with torch.no_grad():
        # Forward pass
        processed_image = processed_image.to(device_type)
        output = model(processed_image.unsqueeze(0))
        
        # Calculate probabilities
        probabilities = torch.exp(output)
        
        # Get the top K probabilities and classes
        top_probabilities, top_classes = probabilities.topk(topk, dim=1)
        
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i.item()] for i in top_classes[0]]

    cat_to_name = load_cat_names()
    class_names = [cat_to_name[str(cls)] for cls in top_classes]
    
    # Convert tensors to lists
    top_probabilities = top_probabilities[0].tolist()
    mydict = {}
    for e, value in enumerate(top_probabilities):
        mydict[f'{class_names[e]}'] = [top_probabilities[e], top_classes[e]]
    
    return mydict


if __name__=="__main__":
    print('\n______________________________________________Starting Prediction Script______________________________________________________________________\n')

    file_name, hidden_layers, network = find_model_file()
    print('Would you like to run the program using GPU or CPU\n')
    device = None
    device_list = ['cpu', 'gpu']
    while device not in device_list:
        device = str(input("\nEnter either 'CPU' or GPU\n\n")).lower()

    if device == 'cpu':
        pass
    elif device == 'gpu':
        device = 'cuda'
    model = load_checkpoint(file_name,hidden_layers,network)
    image_tensor =image_processing('flowers/test/9/image_06413.jpg')
    category_names = load_cat_names()
    mydict = predict(image_tensor, model,device, topk=5)
    print(mydict)

    

