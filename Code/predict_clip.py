import torch
from torchvision import transforms
import clip
from PIL import Image
import numpy as np
from clip_classifier import CLIPClassifier

# Load the CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define transformations for incoming images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

clip_classifier = CLIPClassifier(model, num_classes=2).to(device)

# Load the saved state dict, adjusting keys
state_dict = torch.load('../Models/best_clip_classifier.pth', map_location=device)

# Handle the mismatched keys here
for key in list(state_dict.keys()):
    # Replace fc.weight with classifier.weight and fc.bias with classifier.bias
    if key.startswith('fc.'):
        new_key = key.replace('fc.', 'classifier.')
        state_dict[new_key] = state_dict.pop(key)

# Load corrected state_dict
clip_classifier.load_state_dict(state_dict)
clip_classifier.eval()  # Set model to evaluation mode

# Function to predict with the loaded model
def predict_CLIP(image):
    img = Image.open(image).convert('RGB')
    img = transform(img).unsqueeze(0)
    print('Image recognized and transformed.')

    with torch.no_grad():
        img = img.to(device)
        outputs = clip_classifier(img)
        likelihood_scores = torch.softmax(outputs, dim=1)[0].tolist()
        likelihood_benign = likelihood_scores[0]
        likelihood_malignant = likelihood_scores[1]
        print(f'Likelihood scores: benign: {likelihood_benign:.4f}, malignant: {likelihood_malignant:.4f}')

    return likelihood_benign, likelihood_malignant

    return likelihood_score

