import torch
import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        # Load a pretrained ResNet-101 model
        self.resnet = models.resnet101(weights=True)

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Forward pass through ResNet
        x = self.resnet(x)
        return x

