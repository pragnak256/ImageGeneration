# import modules
import io
import logging

# import third party modules
import torch
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms

def load_model(model_path, labels_path):
    """
    Utility to load the model and class names

    Args:
        model_path (str): Path to the model file
        labels_path (str): Path to the class names file

    Returns:
        Tuple[torch.nn.Module, List[str]]: Model and class names
    """
    model = models.resnet50()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(labels_path) as f:
        model_classes = f.readlines()
    model_classes = [class_name.strip() for class_name in model_classes]

    return model, model_classes

def preprocess_image(image_bytes):
    """
    Utility to preprocess an image for the model

    Args:
        image_bytes (bytes): Image bytes
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logging.debug(f"Image size: {image.size}")
    logging.debug(f"Image mode: {image.mode}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ]) 
    return transform(image).unsqueeze(0)

def predict_image(model, model_classes, image_tensor):
    """
    Utility to predict the top 3 classes for an image using the model

    Args:
        model (torch.nn.Module): Pretrained model
        model_classes (List): List of class names
        image_tensor (torch.Tensor): Image tensor of shape (1, 3, H, W)

    Returns:
        list of tuples: List of top 3 (index, confidence) pairs
    """
    with torch.no_grad():
        # forward pass through the model
        output = model(image_tensor)
        
        # apply softmax to convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        logging.debug(f"Probabilities: {probabilities}")       

        # get the top 3 indices and their probabilities
        probs, indices = torch.topk(probabilities, 3, dim=1)
        logging.debug(f"Top 3 probabilities: {probs}") 
        logging.debug(f"Top 3 indices: {indices}")

        # convert to Python list and return
        results = [
            {"class": model_classes[index.item()], "confidence": int(100*prob.item())} \
                for index, prob in zip(indices[0], probs[0])]
    
    return results

