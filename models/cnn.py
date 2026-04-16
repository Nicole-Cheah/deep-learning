from torchvision import models
import torch.nn as nn

def get_resnet50_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model