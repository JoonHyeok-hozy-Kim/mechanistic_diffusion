import os
import torch
import torch.nn as nn
import torchvision.models as models
from data.emnist.manage_emnist_dataset import MAX_EMNIST_NUM_CLASSES

def get_resnet_emnist():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    num_classes = MAX_EMNIST_NUM_CLASSES    
    
    # Reduce input channel : 3 -> 1
    model.conv1 = nn.Conv2d(
        in_channels=1, 
        out_channels=64, 
        kernel_size=7, 
        stride=2, 
        padding=3, 
        bias=False
    )
    
    # Modify output channel : 1000 -> num_classes (35)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    pre_trained_weight_path = "weights/classifier_weights_reserved/resnet/resnet_emnist_best.pt"
    assert os.path.exists(pre_trained_weight_path), f"Pre-trained weight for the EMNIST resnet classifier is missing. Train it using ./classifier_models/resnet/train_resnet_classifiers.py"
    model.load_state_dict(torch.load(pre_trained_weight_path))
    model.eval()
    
    return model