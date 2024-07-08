import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import time
import sys

# Ignore undefined metric warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Track time, start
start_time = time.time()

# Load the CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", device=device)

# Define data augmentations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]),
}

# Dataset class for melanoma images
class MelanomaDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.classes = ['malignant', 'benign']
        self.image_labels = self._load_labels()

    def _load_labels(self):
        labels = []
        for class_idx, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.image_dir, class_name)
            for fname in os.listdir(class_folder):
                labels.append((os.path.join(class_folder, fname), class_idx))
        return labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Skipping image at {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_labels))


# Paths to the datasets
train_image_dir = r'../Image_data/Data/Train'
validate_image_dir = r'../Image_data/Data/Validation'
test_image_dir = r'../Image_data/Data/Test'

# Load and preprocess datasets
train_dataset = MelanomaDataset(train_image_dir, data_transforms['train'])
validate_dataset = MelanomaDataset(validate_image_dir, data_transforms['val'])
test_dataset = MelanomaDataset(test_image_dir, data_transforms['val'])


# Modify the CLIP model for multi-class classification and fine-tune more layers
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


clip_classifier = CLIPClassifier(model, num_classes=2).to(device)


# Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss


criterion = FocalLoss()

# Optimizer and scheduler
optimizer = optim.Adam(clip_classifier.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dataset) // 64, epochs=10)


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '../Models/best_clip_classifier.pth')
        self.val_loss_min = val_loss


# Create weighted sampler
def create_weighted_sampler(dataset):
    class_counts = np.bincount([label for _, label in dataset.image_labels])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for _, label in dataset.image_labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


train_sampler = create_weighted_sampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4)
validate_dataloader = DataLoader(validate_dataset, batch_size=64, num_workers=4)


# Training and evaluation function
def train_and_evaluate(train_dataloader, validate_dataloader, num_epochs=10):
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_val_acc = 0

    for epoch in range(num_epochs):
        clip_classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = clip_classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Validation step
        clip_classifier.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validate_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = clip_classifier(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(validate_dataloader)
        val_acc = 100 * correct / total
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Check early stopping
        early_stopping(val_loss, clip_classifier)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Save the model with the best validation accuracy
        if val_acc > best_val_acc:
            print(f'Saving model with validation accuracy: {val_acc:.2f}%')
            torch.save(clip_classifier.state_dict(), '../Models/best_clip_classifier.pth')
            best_val_acc = val_acc

        # Adjust learning rate
        scheduler.step()

    print("Finished Training")


# Perform k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Save console output to a text file
log_file = '../Results/Final_results.txt'
with open(log_file, 'w') as f:
    # Redirect stdout to the file
    sys.stdout = f

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}')
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, num_workers=4)
        validate_dataloader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler, num_workers=4)

        train_and_evaluate(train_dataloader, validate_dataloader, num_epochs=10)

    # Reset stdout to default (console)
    sys.stdout = sys.__stdout__

# Evaluate the model on the test dataset
clip_classifier.load_state_dict(torch.load('../Models/best_clip_classifier.pth'))
clip_classifier.eval()
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4):
        images, labels = images.to(device), labels.to(device)
        outputs = clip_classifier(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# End measuring time
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")

test_acc = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

# Append evaluation metrics to the console log file
with open(log_file, 'a') as f:
    print(f'\nTest Accuracy: {test_acc:.2f}%', file=f)
    print(f'Precision: {precision:.2f}', file=f)
    print(f'Recall: {recall:.2f}', file=f)
    print(f'F1 Score: {f1:.2f}', file=f)

print(f'Test Accuracy: {test_acc:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Print location of results file
print(f"Results have been logged to: {os.path.abspath(log_file)}")