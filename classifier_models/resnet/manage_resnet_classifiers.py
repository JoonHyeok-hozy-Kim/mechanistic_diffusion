import os
import torch
import torch.nn as nn
import torchvision.models as models
from data.emnist.manage_emnist_dataset import MAX_EMNIST_NUM_CLASSES
from data.kmnist.manage_kmnist_dataset import MAX_KMNIST_NUM_CLASSES

EMNIST_CLASSIFIER_WEIGHT_PATH = "weights/classifier_weights_reserved/resnet/resnet_emnist_best.pt"
KMNIST_CLASSIFIER_WEIGHT_PATH = "weights/classifier_weights_reserved/resnet/resnet_kmnist_best.pt"


def get_resnet_emnist():
    model = get_raw_resnet18(num_classes=MAX_EMNIST_NUM_CLASSES)
    
    pre_trained_weight_path = EMNIST_CLASSIFIER_WEIGHT_PATH
    assert os.path.exists(pre_trained_weight_path), f"Pre-trained weight for the EMNIST resnet classifier is missing. Train it using ./classifier_models/resnet/train_resnet_classifiers.py"
    model.load_state_dict(torch.load(pre_trained_weight_path))
    model.eval()
    
    return model

def get_resnet_kmnist():
    model = get_raw_resnet18(num_classes=MAX_KMNIST_NUM_CLASSES)
    
    pre_trained_weight_path = KMNIST_CLASSIFIER_WEIGHT_PATH
    assert os.path.exists(pre_trained_weight_path), f"Pre-trained weight for the EMNIST resnet classifier is missing. Train it using ./classifier_models/resnet/train_resnet_classifiers.py"
    model.load_state_dict(torch.load(pre_trained_weight_path))
    model.eval()
    
    return model


def get_raw_resnet18(num_classes):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)    
    
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
    
    return model