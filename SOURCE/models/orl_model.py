import torch
from torchvision import models
import torch.nn as nn

class orlModel(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(orlModel, self).__init__()
        self.model = models.vgg16(num_classes=num_classes)
    def forward(self, x):
        out = self.model(x)
        return out