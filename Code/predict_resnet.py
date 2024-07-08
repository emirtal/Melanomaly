# # predict_resnet.py
# import torch
# from torchvision import transforms
# from PIL import Image
# from ResNet_classifer import ResNetClassifier
#
#
# def predict_resnet(image_file):
#     # Load your pre-trained model
#     model = ResNetClassifier(num_classes=2)
#
#     # Load the state dictionary with strict=False
#     state_dict = torch.load('../Models/best_resnet_classifier.pth', map_location=torch.device('cpu'))
#     # state_dict = torch.load('../Models/best_resnet_classifier.pth')
#     # missing_keys, unexpected_keys = model.lct,state_dict(state_dict, strict=False)
#
#     # Print missing and unexpected keys for debugging
#     # print(f"Missing keys: {missing_keys}")
#     # print(f"Unexpected keys: {unexpected_keys}")
#
#     model.eval()
#
#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     # Preprocess the image
#     image = Image.open(image_file).convert('RGB')
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#
#     # Perform prediction
#     with torch.no_grad():
#         outputs = model(image)
#         _, predicted = torch.max(outputs.data, 1)
#
#     # For demonstration, we assume 0: benign, 1: malignant
#     likelihood_score = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
#
#     return likelihood_score


import torch
from torchvision import transforms
from PIL import Image
from ResNet_classifer import ResNetClassifier

def predict_resnet(image_file):
    # Load your pre-trained model
    model = ResNetClassifier(num_classes=2)

    # Load the state dictionary with strict=False for debugging
    state_dict = torch.load('../Models/best_resnet_classifier.pth', map_location=torch.device('cpu'))

    # Adjust keys in state_dict if necessary
    new_state_dict = {}
    for k, v in state_dict.items():
        # Replace 'conv1.weight' with 'resnet.conv1.weight' and so on...
        if k.startswith('conv1'):
            new_state_dict['resnet.' + k] = v
        elif k.startswith('bn1'):
            new_state_dict['resnet.' + k] = v
        else:
            new_state_dict[k] = v

    # Load state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    # Print missing and unexpected keys for debugging
    missing_keys = []
    unexpected_keys = []
    for key in model.state_dict().keys():
        if key not in new_state_dict:
            missing_keys.append(key)
    for key in new_state_dict.keys():
        if key not in model.state_dict():
            unexpected_keys.append(key)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    model.eval()

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess the image
    image = Image.open(image_file).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    # For demonstration, we assume 0: benign, 1: malignant
    likelihood_score = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()

    return likelihood_score
