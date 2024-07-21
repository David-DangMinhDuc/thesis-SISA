import torch
from torchvision import models
import torch.nn as nn

class arModel(nn.Module):
    def __init__(self, input_shape, num_classes, *args, **kwargs):
        super(arModel, self).__init__()
        aln = models.alexnet(pretrained=True)
        num_ftrs = aln.classifier[6].in_features
        aln.classifier[6] = nn.Linear(num_ftrs, num_classes)
        self.model = aln#models.alexnet(num_classes=num_classes)
    def forward(self, x):
        out = self.model(x)
        return out