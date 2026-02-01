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
