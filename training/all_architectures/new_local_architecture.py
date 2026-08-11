import os
import json
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
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2"

# FEATURES TO COMPARE (Targeting LOCAL folders now)
FEATURES_TO_COMPARE = [  'color', 'brushstroke', 'texture',]

# MODELS TO TRAIN
MODELS_TO_TRAIN = ['resnet50', 'vgg16', 'efficientnet_b3', 'alexnet']

# NEW OUTPUT FOLDERS FOR LOCAL RESULTS
RESULTS_DIR = os.path.join(BASE_DIR, r"results\Local_Architecture_Comparison")
CHECKPOINT_DIR = os.path.join(BASE_DIR, r"checkpoints\local_comparison")

# FAIR COMPARISON SETTINGS
BATCH_SIZE = 64         
NUM_EPOCHS = 20         
NUM_FOLDS = 5           
LEARNING_RATE = 1e-4
NUM_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================================
# 2. UTILS
# ==========================================
def save_json(data, filepath):
    with open(filepath, 'w') as f: json.dump(data, f, indent=4)

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f: return json.load(f)
    return {}

# ==========================================
# 3. MODEL FACTORY
# ==========================================
def initialize_model(model_name, num_classes):
    model = None
    
    if model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    return model.to(device)

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_fold(model_name, feature_type, fold_num, train_idx, val_idx, full_dataset, results_json_path, final_results):
    print(f"\n   >>> {model_name.upper()} ({feature_type}) | FOLD {fold_num}/{NUM_FOLDS}")

    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = initialize_model(model_name, len(full_dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"local_{model_name}_{feature_type}_fold{fold_num}_resume.pth")
    best_model_path = os.path.join(CHECKPOINT_DIR, f"local_{model_name}_{feature_type}_fold{fold_num}_best.pth")

    start_epoch = 0
    best_acc = 0.0
    history = []

    if os.path.exists(ckpt_path):
        print("      [RESUME] Found checkpoint. Loading...")
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        history = checkpoint.get('history', [])

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0; correct = 0; total = 0
        
        loop = tqdm(train_loader, desc=f"      Ep {epoch+1}", leave=False)
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
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0; all_preds = []; all_labels = []
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

        print(f"      Ep {epoch+1}: Val Acc {val_acc:.4f} | F1 {f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_acc": train_acc, "train_loss": avg_train_loss,
            "val_acc": val_acc, "val_loss": avg_val_loss,
            "precision": p, "recall": r, "f1_score": f1
        })

        final_results[f"fold_{fold_num}"] = {
            "status": "in_progress", 
            "best_val_acc": max(best_acc, val_acc), 
            "history": history
        }
        save_json(final_results, results_json_path)

        if val_acc > best_acc: 
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc, 
            'history': history
        }, ckpt_path)

    final_results[f"fold_{fold_num}"] = {"status": "completed", "best_val_acc": best_acc, "history": history}
    save_json(final_results, results_json_path)
    
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    print(f"   >> Fold {fold_num} Done. Best Acc: {best_acc:.4f}")

# ==========================================
# 5. RESULT AGGREGATOR (FIXED)
# ==========================================
def generate_summary_table():
    print("\n[Generating Local Summary CSV...]")
    summary_data = []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            # FIXED PARSING LOGIC HERE
            name_without_ext = filename.replace(".json", "")
            parts = name_without_ext.split("_")
            
            # Expects format: local_{model_name}_{feature}.json
            # e.g., local_efficientnet_b3_color.json
            if len(parts) >= 3:
                feature = parts[-1] 
                model = "_".join(parts[1:-1]) # Joins 'efficientnet_b3' correctly
            
                data = load_json(os.path.join(RESULTS_DIR, filename))
                
                fold_accs = []
                fold_f1s = []
                
                for key, val in data.items():
                    if "fold" in key and val.get("status") == "completed":
                        best_acc = val.get("best_val_acc", 0)
                        best_f1 = 0
                        for h in val.get("history", []):
                            if h["val_acc"] == best_acc:
                                best_f1 = h["f1_score"]
                                break
                        fold_accs.append(best_acc)
                        fold_f1s.append(best_f1)
                
                if fold_accs:
                    summary_data.append({
                        "Model": model,
                        "Feature": feature,
                        "Avg_Accuracy": np.mean(fold_accs),
                        "Avg_F1_Score": np.mean(fold_f1s)
                    })

    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(BASE_DIR, "results\Local_Architecture_Comparison", "Local_Comparison_Table.csv")
        df.sort_values(by=["Avg_Accuracy"], ascending=False, inplace=True)
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")
        print(df)
    else:
        print("No completed results found yet.")

# ==========================================
# 6. MAIN
# ==========================================
def main():
    torch.backends.cudnn.benchmark = True
    print(f"--- STARTING LOCAL ARCHITECTURE COMPARISON ---")
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for feature in FEATURES_TO_COMPARE:
        # POINTING TO LOCAL PROCESSED DATA
        data_dir = os.path.join(BASE_DIR, f"data\\processed\\local_data_{feature}")
        print(f"\n\n{'='*50}\nFEATURE: {feature.upper()}\n{'='*50}")
        
        if not os.path.exists(data_dir):
            print(f"ERROR: Directory not found: {data_dir}")
            print("Did you run the preprocessing scripts with MODE='Local'?")
            continue

        try:
            full_dataset = datasets.ImageFolder(data_dir, transform=tf)
            targets = [s[1] for s in full_dataset.samples]
            skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        except Exception as e:
            print(f"Error loading {feature}: {e}"); continue

        for model_name in MODELS_TO_TRAIN:
            # SAVING AS LOCAL_{MODEL}_{FEATURE}
            results_json_path = os.path.join(RESULTS_DIR, f"local_{model_name}_{feature}.json")
            final_results = load_json(results_json_path)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
                fold_num = fold + 1
                
                if f"fold_{fold_num}" in final_results and final_results[f"fold_{fold_num}"].get("status") == "completed":
                    print(f"   [Skipping] {model_name} ({feature}) Fold {fold_num} done.")
                    continue
                
                train_fold(model_name, feature, fold_num, train_idx, val_idx, full_dataset, results_json_path, final_results)

    generate_summary_table()
    print("\nAll Local Comparisons Completed!")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()