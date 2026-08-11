"""
Quick Parameter Sweep for Global Dataset Preprocessing
Tests different RADIUS/Canny combinations to find optimal settings
Runs 1 fold, 5 epochs per test = ~4 hours total
"""

import os
import json
import subprocess
import time
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

# Configuration
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2"
PREPROCESSING_DIR = os.path.join(BASE_DIR, r"python_codes\preprocessing")
RESULTS_DIR = os.path.join(BASE_DIR, r"results\Parameter_Sweep")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Quick test settings
BATCH_SIZE = 64
NUM_EPOCHS = 5  # Quick validation
NUM_FOLDS = 1   # Only first fold
LEARNING_RATE = 1e-4
NUM_WORKERS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test configurations
TEXTURE_TESTS = [
    {"name": "RADIUS_1", "radius": 1, "n_points": 8},
    {"name": "RADIUS_3", "radius": 3, "n_points": 24}
]

BRUSHSTROKE_TESTS = [
    {"name": "DYNAMIC_CANNY", "use_dynamic": True},
    {"name": "FIXED_CANNY", "use_dynamic": False, "lower": 50, "upper": 150}
]

def modify_preprocessing_script(script_path, mode, params):
    """Modify preprocessing script parameters"""
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Update MODE
    for i, line in enumerate(lines):
        if line.startswith('MODE = '):
            lines[i] = f'MODE = "{mode}"  # Parameter sweep test\n'
    
    # Update feature-specific parameters
    if 'radius' in params:
        # Texture parameters
        for i, line in enumerate(lines):
            if 'if MODE == "Global":' in line and i < len(lines) - 3:
                lines[i+1] = f'    RADIUS = {params["radius"]}      # Sweep test\n'
                lines[i+2] = f'    N_POINTS = {params["n_points"]}\n'
                break
    
    if 'use_dynamic' in params:
        # Brushstroke parameters
        for i, line in enumerate(lines):
            if 'if MODE == "Global":' in line and i < len(lines) - 3:
                lines[i+1] = f'    USE_DYNAMIC_CANNY = {params["use_dynamic"]}  # Sweep test\n'
                if not params['use_dynamic']:
                    lines[i+2] = f'    LOWER_THRESHOLD = {params["lower"]}\n'
                    lines[i+3] = f'    UPPER_THRESHOLD = {params["upper"]}\n'
                break
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def run_preprocessing(script_name):
    """Run preprocessing script"""
    script_path = os.path.join(PREPROCESSING_DIR, script_name)
    print(f"\n{'='*60}\nRunning {script_name}...\n{'='*60}")
    result = subprocess.run(['python', script_path], cwd=PREPROCESSING_DIR, capture_output=False)
    return result.returncode == 0

def initialize_model(num_classes):
    """Initialize EfficientNetB3 (best performer)"""
    model = models.efficientnet_b3(weights='DEFAULT')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model.to(device)

def quick_train(data_dir, test_name):
    """Train for 5 epochs on 1 fold"""
    print(f"\n{'='*60}\nTraining: {test_name}\n{'='*60}")
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(data_dir, transform=tf)
    targets = [s[1] for s in full_dataset.samples]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, val_idx = list(skf.split(np.zeros(len(targets)), targets))[0]
    
    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    model = initialize_model(len(full_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    
    best_acc = 0.0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = correct / total
        avg_train_loss = train_loss / total
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(inputs)
                    l = criterion(outputs, labels)
                
                val_loss += l.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        avg_val_loss = val_loss / len(all_labels)
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        if val_acc > best_acc:
            best_acc = val_acc
        
        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "f1_score": f1
        })
        
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | F1 {f1:.4f}")
    
    return best_acc, history

def main():
    results = {}
    
    print("="*80)
    print("PARAMETER SWEEP FOR GLOBAL DATASET")
    print("Testing optimal preprocessing parameters")
    print("="*80)
    
    # ========== TEXTURE SWEEP ==========
    print("\n\n### TEXTURE PARAMETER SWEEP ###\n")
    
    texture_script = os.path.join(PREPROCESSING_DIR, "preprocess_texture.py")
    texture_results = {}
    
    for test in TEXTURE_TESTS:
        print(f"\n>>> Testing TEXTURE with {test['name']}")
        
        # Modify preprocessing script
        modify_preprocessing_script(texture_script, "Global", {
            "radius": test['radius'],
            "n_points": test['n_points']
        })
        
        # Clean old processed data
        processed_dir = os.path.join(BASE_DIR, r"data\processed\global_data_texture")
        if os.path.exists(processed_dir):
            import shutil
            shutil.rmtree(processed_dir)
        
        # Run preprocessing
        if not run_preprocessing("preprocess_texture.py"):
            print(f"ERROR: Preprocessing failed for {test['name']}")
            continue
        
        # Quick train
        best_acc, history = quick_train(processed_dir, test['name'])
        
        texture_results[test['name']] = {
            "best_acc": best_acc,
            "params": test,
            "history": history
        }
        
        print(f"\n>>> {test['name']} BEST ACCURACY: {best_acc:.4f}\n")
    
    results['texture'] = texture_results
    
    # ========== BRUSHSTROKE SWEEP ==========
    print("\n\n### BRUSHSTROKE PARAMETER SWEEP ###\n")
    
    brushstroke_script = os.path.join(PREPROCESSING_DIR, "preprocess_brushstroke.py")
    brushstroke_results = {}
    
    for test in BRUSHSTROKE_TESTS:
        print(f"\n>>> Testing BRUSHSTROKE with {test['name']}")
        
        # Modify preprocessing script
        params = {"use_dynamic": test['use_dynamic']}
        if 'lower' in test:
            params.update({"lower": test['lower'], "upper": test['upper']})
        
        modify_preprocessing_script(brushstroke_script, "Global", params)
        
        # Clean old processed data
        processed_dir = os.path.join(BASE_DIR, r"data\processed\global_data_brushstroke")
        if os.path.exists(processed_dir):
            import shutil
            shutil.rmtree(processed_dir)
        
        # Run preprocessing
        if not run_preprocessing("preprocess_brushstroke.py"):
            print(f"ERROR: Preprocessing failed for {test['name']}")
            continue
        
        # Quick train
        best_acc, history = quick_train(processed_dir, test['name'])
        
        brushstroke_results[test['name']] = {
            "best_acc": best_acc,
            "params": test,
            "history": history
        }
        
        print(f"\n>>> {test['name']} BEST ACCURACY: {best_acc:.4f}\n")
    
    results['brushstroke'] = brushstroke_results
    
    # ========== SAVE RESULTS ==========
    results_path = os.path.join(RESULTS_DIR, "parameter_sweep_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # ========== SUMMARY ==========
    print("\n\n" + "="*80)
    print("PARAMETER SWEEP RESULTS")
    print("="*80)
    
    print("\nTEXTURE:")
    for name, data in texture_results.items():
        print(f"  {name}: {data['best_acc']:.4f}")
    
    print("\nBRUSHSTROKE:")
    for name, data in brushstroke_results.items():
        print(f"  {name}: {data['best_acc']:.4f}")
    
    # Recommend best
    best_texture = max(texture_results.items(), key=lambda x: x[1]['best_acc'])
    best_brushstroke = max(brushstroke_results.items(), key=lambda x: x[1]['best_acc'])
    
    print("\n" + "="*80)
    print("RECOMMENDED PARAMETERS:")
    print("="*80)
    print(f"Texture: {best_texture[0]} (Accuracy: {best_texture[1]['best_acc']:.4f})")
    print(f"  Parameters: {best_texture[1]['params']}")
    print(f"\nBrushstroke: {best_brushstroke[0]} (Accuracy: {best_brushstroke[1]['best_acc']:.4f})")
    print(f"  Parameters: {best_brushstroke[1]['params']}")
    print("="*80)
    
    print(f"\nFull results saved to: {results_path}")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
