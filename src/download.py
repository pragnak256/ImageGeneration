# import modules
import os
import requests

# import thirdy party modules
import torch

# specify the path to download the model and labels file
model_path = 'models/'
model_file_path = os.path.join(model_path, 'resnet50.pth')
os.makedirs(model_path, exist_ok=rTrue)

# load the ResNet50 model
model_url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
torch.hub.download_url_to_file(model_url, model_file_path)

# download the ImageNet class labels to models path
labels_file_path = os.path.join(model_path, 'imagenet_classes.txt')
labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'

response = requests.get(labels_url)
with open(labels_file_path, 'w') as file:
    file.write(response.text)