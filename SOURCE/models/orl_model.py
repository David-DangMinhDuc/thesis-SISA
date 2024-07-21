import torch
from torchvision import models
import torch.nn as nn

class orlModel(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(orlModel, self).__init__()
        vgg_16 = models.vgg16(pretrained=True)
        num_ftrs = vgg_16.classifier[6].in_features
        vgg_16.classifier[6] = nn.Linear(num_ftrs, num_classes)
        self.model = vgg_16#models.vgg16(num_classes=num_classes)
    def forward(self, x):
        out = self.model(x)
        return out