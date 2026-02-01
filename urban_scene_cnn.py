"""
Week 3 Assignment: Neural Network Models for Urban Scene Classification
Using VS Code, Python, and GitHub Copilot

This script is organized into steps matching the assignment document.
Commit after completing each step as indicated.

https://www.kaggle.com/datasets/mittalshubham/images256
"""

# =============================================================================
# STEP 3: Load and Prepare the Dataset (MIT Places Subset)
# =============================================================================

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import kagglehub
import os
import shutil
from pathlib import Path

print("="*70)
print("STEP 3: Loading MIT Places Dataset from Kaggle")
print("="*70)

# Download dataset using KaggleHub (streams from Kaggle, no large local download)
print("\n📥 Accessing MIT Places dataset from Kaggle...")
print("   (Using KaggleHub - data is streamed, not fully downloaded)")

try:
    # Download the dataset - KaggleHub caches it
    # Updated dataset source
    dataset_download_path = kagglehub.dataset_download("mittalshubham/images256")
    print(f"✅ Dataset accessed at: {dataset_download_path}")
    
    # The new dataset structure:
    # - a/
    #   - airfield/
    #   - alcove/
    #   - ...
    # - b/
    # - ...
    # - z/
    
    print(f"\n📂 Dataset location: {dataset_download_path}")
    
    # Define urban categories we want to focus on
    urban_categories = [
        'street',
        'highway',
        'skyscraper',
        'office_building',
        'downtown',
        'crosswalk',
        'parking_lot',
        'bridge',
        'building_facade',
        'apartment_building'
    ]
    
    # Create a temporary directory with only urban categories
    urban_dataset_path = "./MIT_Places_Urban_Subset"
    
    if not os.path.exists(urban_dataset_path):
        print(f"\n🏗️ Creating urban subset at: {urban_dataset_path}")
        os.makedirs(urban_dataset_path, exist_ok=True)
        
        # Copy only urban categories
        for category in urban_categories:
            # Find the category in the dataset structure
            # Categories are organized by first letter in subfolders (a/, b/, c/, etc.)
            first_letter = category[0].lower()
            source_category_path = os.path.join(dataset_download_path, first_letter, category)
            
            if os.path.exists(source_category_path):
                dest_category_path = os.path.join(urban_dataset_path, category)
                
                # Create symbolic link instead of copying to save space
                if not os.path.exists(dest_category_path):
                    try:
                        os.symlink(source_category_path, dest_category_path)
                        num_images = len([f for f in os.listdir(source_category_path) if f.endswith('.jpg')])
                        print(f"   ✅ Linked {category}: {num_images} images")
                    except (OSError, NotImplementedError):
                        # If symlink fails (Windows), copy a subset of images
                        os.makedirs(dest_category_path, exist_ok=True)
                        images = [f for f in os.listdir(source_category_path) if f.endswith('.jpg')][:500]  # Limit to 500 images
                        for img in images:
                            shutil.copy2(
                                os.path.join(source_category_path, img),
                                os.path.join(dest_category_path, img)
                            )
                        print(f"   ✅ Copied {category}: {len(images)} images")
            else:
                print(f"   ⚠️ Category not found: {category}")
    
    dataset_path = urban_dataset_path
    
except Exception as e:
    print(f"❌ Error accessing Kaggle dataset: {e}")
    print("\n💡 Make sure you have:")
    print("   1. Installed kagglehub: pip install kagglehub")
    print("   2. Set up Kaggle credentials in ~/.kaggle/kaggle.json")
    print("   3. Accepted the dataset terms on Kaggle website")
    print("\nUsing placeholder path for now...")
    dataset_path = "./MIT_Places_Urban_Subset"

# Define dataset transformations
print("\n🔧 Setting up data transformations...")
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset from directory
print(f"\n📦 Loading dataset from: {dataset_path}")
try:
    dataset = ImageFolder(root=dataset_path, transform=transform)
    print(f"✅ Dataset loaded successfully!")
    print(f"   Total images: {len(dataset)}")
    print(f"   Number of classes: {len(dataset.classes)}")
    print(f"   Classes: {dataset.classes}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("   Please ensure the dataset path is correct and contains image folders")
    # Create a fallback empty dataset structure
    os.makedirs(dataset_path, exist_ok=True)
    raise

print("\n" + "="*70)
print("✅ STEP 3 COMPLETE: Dataset loaded and preprocessed")
print("="*70)

# Split dataset into training, validation, and test sets
print("\n📊 Splitting dataset...")
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

print(f"   Training set: {len(train_set)} images")
print(f"   Validation set: {len(val_set)} images")
print(f"   Test set: {len(test_set)} images")

# Create data loaders
print("\n🔄 Creating data loaders...")
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(f"✅ Data loaders created")
print(f"   Batch size: 32")
print(f"   Training batches: {len(train_loader)}")
print(f"   Validation batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Display a sample image
print("\n🖼️ Displaying sample image...")
try:
    sample_image, sample_label = dataset[0]
    
    # Denormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    sample_image_display = sample_image * std + mean
    sample_image_display = torch.clamp(sample_image_display, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image_display.permute(1, 2, 0))
    plt.title(f"Sample Image - Class: {dataset.classes[sample_label]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("sample_image.png")
    print("✅ Sample image saved as 'sample_image.png'")
    # plt.show()  # Uncomment to display
except Exception as e:
    print(f"⚠️ Could not display sample image: {e}")

print("\n" + "="*70)
print("✅ STEP 3 COMPLETE: Dataset loaded and preprocessed")
print("="*70)
print("\n💾 COMMIT NOW: 'Loaded and preprocessed MIT Places dataset'\n")


# =============================================================================
# STEP 4: Build a Simple CNN Model
# =============================================================================

import torch.nn as nn
import torch.optim as optim

print("\n" + "="*70)
print("STEP 4: Building CNN Model")
print("="*70)

# Define a simple CNN architecture
class UrbanSceneCNN(nn.Module):
    def __init__(self, num_classes):
        super(UrbanSceneCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        # After one pooling layer with 128x128 input: 128/2 = 64
        # So output size is: 32 channels * 64 * 64 = 131,072
        self.fc1 = nn.Linear(32 * 64 * 64, num_classes)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected
        x = self.fc1(x)
        
        return x

# Initialize model
num_classes = len(dataset.classes)
model = UrbanSceneCNN(num_classes)

print("\n🤖 Model Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Number of classes: {num_classes}")

print("\n" + "="*70)
print("✅ STEP 4 COMPLETE: CNN model implemented")
print("="*70)
print("\n💾 COMMIT NOW: 'Implemented CNN model for urban scene classification'\n")


# =============================================================================
# STEP 5: Train the CNN Model
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Training CNN Model")
print("="*70)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n⚙️ Training Configuration:")
print(f"   Loss function: Cross Entropy Loss")
print(f"   Optimizer: Adam")
print(f"   Learning rate: 0.001")
print(f"   Epochs: 5")

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5):
    """
    Train the CNN model
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimization algorithm
        criterion: Loss function
        epochs: Number of training epochs
    """
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / batch_count
        training_history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Training Loss: {avg_train_loss:.4f}")
        print(f"   Validation Loss: {avg_val_loss:.4f}")
        print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print("-" * 70)
    
    print("\n✅ Training complete!")
    return training_history

# Train model
training_history = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5)

# Plot training history
print("\n📈 Generating training plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
epochs_range = range(1, len(training_history['train_loss']) + 1)
ax1.plot(epochs_range, training_history['train_loss'], 'b-', label='Training Loss')
ax1.plot(epochs_range, training_history['val_loss'], 'r-', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot accuracy
ax2.plot(epochs_range, training_history['val_accuracy'], 'g-', label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
print("✅ Training plots saved as 'training_history.png'")
# plt.show()  # Uncomment to display

print("\n" + "="*70)
print("✅ STEP 5 COMPLETE: Model training finished")
print("="*70)
print("\n💾 COMMIT NOW: 'Trained CNN model for urban scene classification'\n")


# =============================================================================
# STEP 6: Evaluate Model Performance
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Evaluating Model Performance")
print("="*70)

# Evaluate on test data
def evaluate_model(model, test_loader):
    """
    Evaluate the trained model on test data
    
    Args:
        model: The trained neural network model
        test_loader: DataLoader for test data
    
    Returns:
        test_accuracy: Accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0
    
    class_correct = {}
    class_total = {}
    
    print("\n🧪 Evaluating on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                label_name = dataset.classes[label]
                if label_name not in class_correct:
                    class_correct[label_name] = 0
                    class_total[label_name] = 0
                
                class_total[label_name] += 1
                if label == prediction:
                    class_correct[label_name] += 1
    
    test_accuracy = correct / total
    
    print(f"\n📊 Overall Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   Correctly classified: {correct}/{total} images")
    
    # Per-class accuracy
    print("\n📋 Per-Class Accuracy:")
    for class_name in sorted(class_correct.keys()):
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name]
            print(f"   {class_name}: {acc:.4f} ({acc*100:.2f}%) "
                  f"[{class_correct[class_name]}/{class_total[class_name]}]")
    
    return test_accuracy

test_accuracy = evaluate_model(model, test_loader)

# Plot results
print("\n📊 Generating performance visualization...")
plt.figure(figsize=(8, 6))
plt.bar(["Test Accuracy"], [test_accuracy], color='steelblue')
plt.ylabel("Accuracy")
plt.title("CNN Model Performance on Test Set")
plt.ylim([0, 1])
plt.grid(axis='y', alpha=0.3)

# Add value on top of bar
plt.text(0, test_accuracy + 0.02, f'{test_accuracy:.4f}\n({test_accuracy*100:.2f}%)', 
         ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("test_accuracy.png")
print("✅ Performance plot saved as 'test_accuracy.png'")
# plt.show()  # Uncomment to display

# Save the model
print("\n💾 Saving trained model...")
torch.save(model.state_dict(), 'urban_scene_cnn_model.pth')
print("✅ Model saved as 'urban_scene_cnn_model.pth'")

print("\n" + "="*70)
print("✅ STEP 6 COMPLETE: Model evaluation finished")
print("="*70)
print("\n💾 COMMIT NOW: 'Evaluated CNN model performance on test data'\n")

# =============================================================================
# FINAL SUMMARY
# =============================================================================


print("\n📋 Summary:")
print(f"   Dataset: MIT Places (Urban Subset)")
print(f"   Total images: {len(dataset)}")
print(f"   Number of classes: {num_classes}")
print(f"   Model: Simple CNN with Batch Normalization")
print(f"   Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
