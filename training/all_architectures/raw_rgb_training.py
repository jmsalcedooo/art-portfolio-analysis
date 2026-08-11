"""
ALTERNATIVE APPROACH: Train CNNs directly on RAW RGB images
instead of preprocessed features (LBP, Canny, CLAHE)

This lets the CNN learn features automatically rather than relying on handcrafted features.
Expected performance: 80-85%+ for 50-class problem based on literature.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2"

# Use ORIGINAL IMAGES (not preprocessed)
DATA_DIR = os.path.join(BASE_DIR, r"data\global_data")  # Original RGB images

RESULTS_DIR = os.path.join(BASE_DIR, r"results\RAW_RGB_Training")
CHECKPOINT_DIR = os.path.join(BASE_DIR, r"checkpoints\raw_rgb")

# Models to test
MODELS_TO_TRAIN = ['resnet50', 'efficientnet_b3', 'vgg16']

# Training settings
BATCH_SIZE = 32         # Smaller batch for full RGB images
NUM_EPOCHS = 30         # More epochs for deep feature learning
NUM_FOLDS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# DATA AUGMENTATION (Standard ImageNet-style)
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# MODEL BUILDER
# ==========================================
def build_model(model_name, num_classes):
    """Build pretrained CNN and replace final layer"""
    
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'alexnet':
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model.to(device)

# ==========================================
# TRAINING & EVALUATION
# ==========================================
def train_epoch(model, loader, criterion, optimizer, scaler):
    """Train one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    """Validate and return metrics"""
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ==========================================
# MAIN TRAINING LOOP
# ==========================================
def train_model(model_name):
    """Train single model with 5-fold CV"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()} ON RAW RGB IMAGES")
    print(f"{'='*80}")
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(DATA_DIR)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across {num_classes} artists")
    
    # Extract labels for stratification
    labels = [label for _, label in full_dataset.samples]
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    results = {}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n   >>> FOLD {fold}/{NUM_FOLDS}")
        
        # Create fold datasets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # Apply transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        
        # Create loaders
        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Build model
        model = build_model(model_name, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = GradScaler()
        
        # Training loop
        fold_results = {'history': [], 'best_val_acc': 0}
        
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion)
            
            # Save results
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1']
            }
            fold_results['history'].append(epoch_data)
            
            # Update best
            if val_metrics['accuracy'] > fold_results['best_val_acc']:
                fold_results['best_val_acc'] = val_metrics['accuracy']
                
                # Save checkpoint
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, f"{model_name}_fold{fold}_best.pth"
                )
                torch.save(model.state_dict(), checkpoint_path)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        fold_results['status'] = 'completed'
        results[f'fold_{fold}'] = fold_results
    
    # Save results
    result_path = os.path.join(RESULTS_DIR, f"raw_rgb_{model_name}.json")
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Calculate average
    avg_acc = np.mean([results[f'fold_{i}']['best_val_acc'] for i in range(1, NUM_FOLDS+1)])
    print(f"\n✅ {model_name.upper()} Average Accuracy: {avg_acc:.4f}")
    
    return avg_acc

# ==========================================
# RUN ALL MODELS
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALTERNATIVE APPROACH: RAW RGB IMAGE TRAINING")
    print("Training CNNs on original artwork (not handcrafted features)")
    print("="*80)
    
    summary = {}
    for model_name in MODELS_TO_TRAIN:
        avg_acc = train_model(model_name)
        summary[model_name] = avg_acc
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for model_name, acc in summary.items():
        print(f"{model_name}: {acc:.4f}")
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "raw_rgb_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
