import torch
from ResNet_classifer import ResNetClassifier  # Replace with your actual ResNet classifier class

# Load your model
model = ResNetClassifier(num_classes=2)  # Adjust num_classes as needed

# Load the state dictionary (example path)
state_dict = torch.load('../Models/best_resnet_classifier.pth', map_location=torch.device('cpu'))

# Load state_dict into the model (strict=False if necessary)
model.load_state_dict(state_dict)

# Print out the keys of the current state_dict
print("Current state_dict keys:")
for key in model.state_dict().keys():
    print(key)
