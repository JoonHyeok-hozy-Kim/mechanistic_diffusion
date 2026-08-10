import torch.nn as nn
import torch.nn.functional as F

from data.two_dimensional.manage_two_dimensional_data import squeeze_input_data, determine_classes


class Two_Dimensional_Data_Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, x):
        x_sqzd = squeeze_input_data(x)
        pred = determine_classes(x_sqzd, self.num_classes)
        pred_logits = F.one_hot(pred, num_classes=self.num_classes).float()
        pred_logits = pred_logits * 100.0   # For 100% confidence
        
        return pred_logits