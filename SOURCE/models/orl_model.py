import torch
from torchvision import models
import torch.nn as nn

class orlModel(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(orlModel, self).__init__()
        orl_vgg_16 = models.vgg16(pretrained=True)
        orl_vgg_16.classifier[6] = nn.Linear(4096, num_classes)
        self.model = orl_vgg_16
    def forward(self, x):
        out = self.model(x)
        return out