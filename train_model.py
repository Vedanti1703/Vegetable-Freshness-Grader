import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════
DATASET_PATH = "dataset/Train"  # Update this to your dataset path
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_WORKERS = 0  # Set to 0 for Windows, can be >0 for Linux/Mac

# ════════════════════════════════════════════════════════════════
# DATA TRANSFORMATIONS
# ════════════════════════════════════════════════════════════════
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ════════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════════
print("Loading dataset...")
train_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=False 
)

# ════════════════════════════════════════════════════════════════
# CREATE MODEL
# ════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use ResNet18 pretrained on ImageNet
model = models.resnet18(pretrained=True)

# Freeze early layers to speed up training
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last few layers for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Modify final layer for binary classification (fresh vs rotten)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)

model = model.to(device)

# ════════════════════════════════════════════════════════════════
# LOSS AND OPTIMIZER
# ════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════
print(f"\nDataset Info:")
print(f"  Classes: {train_dataset.classes}")
print(f"  Total images: {len(train_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")
print(f"\nStarting training for {EPOCHS} epochs...\n")

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * correct / total
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
    print(f"  Loss: {epoch_loss:.4f}")
    print(f"  Accuracy: {epoch_acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Learning rate scheduling
    scheduler.step(epoch_loss)
    
    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': epoch_acc,
            'class_names': train_dataset.classes
        }, 'freshness_model.pth')
        print(f"✅ New best model saved! (Accuracy: {best_acc:.2f}%)\n")

# ════════════════════════════════════════════════════════════════
# TRAINING COMPLETE
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("🎉 Training Complete!")
print(f"{'='*60}")
print(f"Best Accuracy: {best_acc:.2f}%")
print(f"Model saved as: freshness_model.pth")
print(f"Device used: {device}")
print(f"Total epochs: {EPOCHS}")
print(f"{'='*60}\n")
