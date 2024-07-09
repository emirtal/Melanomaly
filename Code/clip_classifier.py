import torchvision.models as models
import torch
import torch.nn as nn
import clip


class CLIPClassifier(nn.Module):
    def __init__(self, model, num_classes=2):
        super(CLIPClassifier, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True  # Fine-tune all layers
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x)
        features = features.float()  # Convert features to float
        logits = self.classifier(features)
        return logits
