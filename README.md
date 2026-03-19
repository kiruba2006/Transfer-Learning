# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Image classification from scratch requires a huge dataset and long training times. To overcome this, transfer learning can be applied using pre-trained models like VGG-19, which has already learned feature representations from a large dataset (ImageNet).

Problem Statement: Build an image classifier using VGG-19 pre-trained architecture, fine-tuned for a custom dataset (e.g., CIFAR-10, Flowers dataset, or any small image dataset).
Dataset: A dataset consisting of multiple image classes (e.g., train, test, and validation sets). For example, CIFAR-10 (10 classes of small images) or a custom dataset with multiple classes.

## DESIGN STEPS
### STEP 1:
Import the required libraries (PyTorch, torchvision, matplotlib, etc.) and set up the device (CPU/GPU).

### STEP 2:
Load the dataset (train and test). Apply transformations such as resizing, normalization, and augmentation. Create DataLoader objects.

### STEP 3:
Load the pre-trained VGG-19 model from torchvision.models. Modify the final fully connected layer to match the number of classes in the dataset.

### STEP 4:
Define the loss function (CrossEntropyLoss) and the optimizer (Adam).

### STEP 5:
Train the model for the required number of epochs while recording training loss and validation loss.

### STEP 6:
Evaluate the model using a confusion matrix, classification report, and test it on new samples.

## PROGRAM

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset (example: CIFAR-10 or custom dataset)
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(pretrained=True)

# Freeze all layers except classifier
for param in model.features.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader, test_loader, num_epochs=5):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss/len(test_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=10)


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="965" height="815" alt="image" src="https://github.com/user-attachments/assets/45a65f6a-c8e9-4ad6-bf11-f03a7306b859" />

### Confusion Matrix

<img width="908" height="769" alt="image" src="https://github.com/user-attachments/assets/17226f63-f3cd-46ae-aa7a-b6e9d405b8be" />

### Classification Report

<img width="750" height="217" alt="image" src="https://github.com/user-attachments/assets/b9002908-56c1-4c14-92a1-19597eb3b061" />

### New Sample Prediction

<img width="556" height="548" alt="image" src="https://github.com/user-attachments/assets/7e5469fa-8d32-40b2-a002-aa883fc76205" />

<img width="608" height="559" alt="image" src="https://github.com/user-attachments/assets/291dad28-7d31-4736-a237-5bd99c3054e1" />


## RESULT:

Thus to Implement Transfer Learning for classification using VGG-19 architecture is done successfully.
