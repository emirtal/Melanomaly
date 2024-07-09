import torch
from torchvision import transforms
from PIL import Image
from ResNet_classifer import ResNetClassifier
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import sklearn.metrics as metrics

def add_prefix_to_state_dict(state_dict, prefix="resnet."):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = prefix + key
        new_state_dict[new_key] = value
    return new_state_dict

def predict_resnet(image_file):
    # Load your pre-trained model
    model = ResNetClassifier(num_classes=2)

    # Load the state dictionary
    state_dict = torch.load('../Models/best_resnet_classifier.pth', map_location=torch.device('cpu'))

    # Add the "resnet." prefix to each key
    new_state_dict = add_prefix_to_state_dict(state_dict)

    # Load state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)

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
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        likelihood_scores = torch.nn.functional.softmax(outputs, dim=1)[0].tolist()
        likelihood_benign = likelihood_scores[0]
        likelihood_malignant = likelihood_scores[1]

    # Specify the correct layer path for Grad-CAM
    target_layer = 'resnet.layer4'  # Update this path as needed, may be different for you depending on keys!

    # Generate Grad-CAM heatmap
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Forward pass through the model to register hooks
    outputs = model(image_tensor)
    activation_map = cam_extractor(outputs.squeeze(0).argmax().item(), outputs)

    # Overlay heatmap on image
    heatmap = activation_map[0].squeeze().cpu().numpy()
    overlay_image = overlay_mask(transforms.ToPILImage()(image_tensor.squeeze()), transforms.ToPILImage()(heatmap), alpha=0.5)

    # Save heatmap image
    heatmap_path = '../Results/grad_cam_result.jpg'
    overlay_image.save(heatmap_path)

    return likelihood_benign, likelihood_malignant, heatmap_path


