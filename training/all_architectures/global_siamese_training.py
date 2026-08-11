import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2"

# DEFINING THE CHAMPIONS (Based on architecture comparison results)
# Using EfficientNetB3 for all features (best overall performance)
CHAMPIONS = {
    'color': 'efficientnet_b3',
    'texture': 'efficientnet_b3',
    'brushstroke': 'efficientnet_b3'
}

# Training Settings
BATCH_SIZE = 16        
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5   
MARGIN = 1.0           

SIAMESE_CHECKPOINT_DIR = os.path.join(BASE_DIR, r"checkpoints\siamese")
CLASSIFICATION_CHECKPOINT_DIR = os.path.join(BASE_DIR, r"checkpoints\comparison")  # Global checkpoints
LOCAL_CHECKPOINT_DIR = os.path.join(BASE_DIR, r"checkpoints\local_comparison")  # Local checkpoints
RESULTS_DIR = os.path.join(BASE_DIR, r"results\Siamese_Training")

os.makedirs(SIAMESE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# 3. DATASET (TRIPLET GENERATOR - GLOBAL + LOCAL)
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, dataset):
        """Dataset for triplet generation from COMBINED global + local data"""
        self.dataset = dataset
        self.labels = np.array([s[1] for s in dataset.samples])
        self.data = dataset.samples
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

    def __getitem__(self, index):
        img1_path, label1 = self.data[index]
        
        # Positive (Same Class)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        img2_path, _ = self.data[positive_index]
        
        # Negative (Different Class)
        label2 = np.random.choice(list(self.labels_set - {label1}))
        negative_index = np.random.choice(self.label_to_indices[label2])
        img3_path, _ = self.data[negative_index]

        img1 = self.dataset.loader(img1_path)
        img2 = self.dataset.loader(img2_path)
        img3 = self.dataset.loader(img3_path)

        if self.dataset.transform is not None:
            img1 = self.dataset.transform(img1)
            img2 = self.dataset.transform(img2)
            img3 = self.dataset.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.dataset)

# ==========================================
# 4. MODEL BUILDER
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base = base_model
        
        # Replace final layer with Embedding Layer (128-dim)
        if hasattr(self.base, 'fc'): # ResNet
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
        elif hasattr(self.base, 'classifier'): # VGG/EfficientNet
            if isinstance(self.base.classifier, nn.Sequential):
                last_layer_idx = len(self.base.classifier) - 1
                num_ftrs = self.base.classifier[last_layer_idx].in_features
                self.base.classifier[last_layer_idx] = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
    
    def forward(self, x):
        return self.base(x)

def load_pretrained_champion(feature, model_name, dataset_type='global', num_classes=50):
    print(f"   Finding Best Pre-trained {dataset_type.upper()} Weights for {model_name.upper()} ({feature})...")
    
    # 1. Initialize Model Architecture with correct num_classes
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes) 
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # 2. SMART SEARCH: Read JSON to find the best fold
    if dataset_type == 'global':
        json_candidates = [
            os.path.join(BASE_DIR, r"results\Architecture_Comparison", f"global_{model_name}_{feature}.json"),
            os.path.join(BASE_DIR, r"results\Global_Architecture_Comparison", f"global_{model_name}_{feature}.json")
        ]
    else:  # local
        json_candidates = [
            os.path.join(BASE_DIR, r"results\Local_Architecture_Comparison", f"local_{model_name}_{feature}.json")
        ]
    
    best_fold = 1
    highest_acc = 0.0
    json_found = False

    for json_path in json_candidates:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    for fold_key, fold_data in data.items():
                        if "fold_" in fold_key and fold_data.get("status") == "completed":
                            acc = fold_data.get("best_val_acc", 0.0)
                            if acc > highest_acc:
                                highest_acc = acc
                                best_fold = int(fold_key.split("_")[1])
                print(f"      [Smart Select] Best performance found: Fold {best_fold} (Acc: {highest_acc:.4f})")
                json_found = True
                break
            except Exception:
                continue
    
    if not json_found:
        print(f"      [Warning] JSON log not found. Defaulting to Fold 1.")

    # 3. Construct Path to that specific fold
    if dataset_type == 'global':
        possible_paths = [
            # Global checkpoints saved WITHOUT "global_" prefix
            os.path.join(CLASSIFICATION_CHECKPOINT_DIR, f"{model_name}_{feature}_fold{best_fold}_best.pth"),
            # Fallback to local if global doesn't exist (e.g., color was only trained locally)
            os.path.join(LOCAL_CHECKPOINT_DIR, f"local_{model_name}_{feature}_fold{best_fold}_best.pth")
        ]
    else:  # local
        possible_paths = [
            os.path.join(LOCAL_CHECKPOINT_DIR, f"local_{model_name}_{feature}_fold{best_fold}_best.pth")
        ]
    
    loaded = False
    for ckpt_path in possible_paths:
        if os.path.exists(ckpt_path):
            checkpoint_type = "LOCAL" if "local_" in os.path.basename(ckpt_path) else "GLOBAL"
            print(f"      ✅ Loading {checkpoint_type} weights: {os.path.basename(ckpt_path)}")
            checkpoint = torch.load(ckpt_path)
            
            # Robust loading (Handles Dictionary vs Raw Weights)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            loaded = True
            break
    
    if not loaded:
        print(f"      ⚠️  [FALLBACK] Fold {best_fold} checkpoint not found. Using ImageNet pretrained weights.")
        if model_name == 'resnet50': model = models.resnet50(weights='DEFAULT')
        elif model_name == 'efficientnet_b3': model = models.efficientnet_b3(weights='DEFAULT')

    return model
# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_siamese(feature, model_name):
    print(f"\n{'#'*60}")
    print(f"STARTING SIAMESE TRAINING: {feature.upper()} (Model: {model_name})")
    print(f"{'#'*60}")
    
    # Paths
    results_json_path = os.path.join(RESULTS_DIR, f"siamese_{model_name}_{feature}.json")
    ckpt_path = os.path.join(SIAMESE_CHECKPOINT_DIR, f"siamese_{feature}_resume.pth")
    best_model_path = os.path.join(SIAMESE_CHECKPOINT_DIR, f"siamese_{feature}_best.pth")

    # Load Previous Results
    final_results = load_json(results_json_path)
    if not final_results:
        final_results = {"status": "in_progress", "best_triplet_acc": 0.0, "history": []}

    # Data - COMBINE GLOBAL + LOCAL datasets
    global_dir = os.path.join(BASE_DIR, f"data\\processed\\global_data_{feature}")
    local_dir = os.path.join(BASE_DIR, f"data\\processed\\local_data_{feature}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load both datasets
    global_dataset = datasets.ImageFolder(global_dir, transform=transform)
    local_dataset = datasets.ImageFolder(local_dir, transform=transform)
    
    # Combine datasets (shift local labels by global class count)
    num_global_classes = len(global_dataset.classes)
    num_local_classes = len(local_dataset.classes)
    
    # Adjust local labels to avoid overlap
    combined_samples = global_dataset.samples.copy()
    for path, label in local_dataset.samples:
        combined_samples.append((path, label + num_global_classes))
    
    # Create combined dataset
    combined_dataset = datasets.ImageFolder(global_dir, transform=transform)
    combined_dataset.samples = combined_samples
    combined_dataset.targets = [s[1] for s in combined_samples]
    combined_dataset.classes = global_dataset.classes + local_dataset.classes
    
    print(f"   Combined Dataset: {num_global_classes} global + {num_local_classes} local = {len(combined_dataset.classes)} total classes")
    print(f"   Total images: {len(combined_dataset)} ({len(global_dataset)} global + {len(local_dataset)} local)")
    
    triplet_dataset = TripletDataset(combined_dataset)
    dataloader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model - Load GLOBAL champion (will be adapted for Siamese embedding)
    base_model = load_pretrained_champion(feature, model_name, dataset_type='global', num_classes=num_global_classes)
    siamese_net = SiameseNetwork(base_model).to(device)
    
    criterion = nn.TripletMarginLoss(margin=MARGIN)
    optimizer = optim.Adam(siamese_net.parameters(), lr=LEARNING_RATE)
    
    # Resume Logic
    start_epoch = 0
    best_triplet_acc = final_results.get("best_triplet_acc", 0.0)
    
    if os.path.exists(ckpt_path):
        print("      [RESUME] Found checkpoint. Loading...")
        checkpoint = torch.load(ckpt_path)
        siamese_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_triplet_acc = checkpoint['best_triplet_acc']

    # Epoch Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        siamese_net.train()
        running_loss = 0.0
        correct_triplets = 0
        total_triplets = 0
        total_pos_dist = 0.0
        total_neg_dist = 0.0
        
        loop = tqdm(dataloader, desc=f"   Ep {epoch+1}/{NUM_EPOCHS}", leave=False)
        for anchor, positive, negative in loop:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            emb_a = siamese_net(anchor)
            emb_p = siamese_net(positive)
            emb_n = siamese_net(negative)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            
            # --- CALCULATE METRICS ---
            # Euclidean Distance
            dist_pos = F.pairwise_distance(emb_a, emb_p)
            dist_neg = F.pairwise_distance(emb_a, emb_n)
            
            # Accuracy: Is Positive closer than Negative?
            is_correct = (dist_pos < dist_neg).float()
            correct_triplets += is_correct.sum().item()
            total_triplets += anchor.size(0)
            
            total_pos_dist += dist_pos.mean().item()
            total_neg_dist += dist_neg.mean().item()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Averages
        avg_loss = running_loss / len(dataloader)
        triplet_acc = correct_triplets / total_triplets
        avg_pos = total_pos_dist / len(dataloader)
        avg_neg = total_neg_dist / len(dataloader)
        
        print(f"   Ep {epoch+1}: Loss {avg_loss:.4f} | TripAcc {triplet_acc:.4f} | PosDist {avg_pos:.3f} | NegDist {avg_neg:.3f}")
        
        # Update History
        final_results["history"].append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "triplet_acc": triplet_acc,
            "avg_pos_dist": avg_pos,
            "avg_neg_dist": avg_neg
        })
        final_results["best_triplet_acc"] = max(best_triplet_acc, triplet_acc)
        save_json(final_results, results_json_path)

        # Save Best Model (Based on Triplet Accuracy)
        if triplet_acc > best_triplet_acc:
            best_triplet_acc = triplet_acc
            torch.save(siamese_net.state_dict(), best_model_path)
            
        # Save Resume Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': siamese_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_triplet_acc': best_triplet_acc
        }, ckpt_path)

    final_results["status"] = "completed"
    save_json(final_results, results_json_path)
    if os.path.exists(ckpt_path): os.remove(ckpt_path)
    print(f"   >> Training Complete for {feature}. Best Acc: {best_triplet_acc:.4f}")

# ==========================================
# 6. RESULT SUMMARY
# ==========================================
def generate_summary():
    print("\n[Generating Siamese Summary CSV...]")
    summary_data = []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith(".json"):
            data = load_json(os.path.join(RESULTS_DIR, filename))
            
            # Filename format: siamese_{model}_{feature}.json
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 3:
                feature = parts[-1]
                model = "_".join(parts[1:-1])
                
                best_acc = data.get("best_triplet_acc", 0)
                
                # Get final metrics
                if data.get("history"):
                    last_epoch = data["history"][-1]
                    final_loss = last_epoch["loss"]
                    avg_pos = last_epoch["avg_pos_dist"]
                    avg_neg = last_epoch["avg_neg_dist"]
                else:
                    final_loss, avg_pos, avg_neg = 0,0,0
                
                summary_data.append({
                    "Feature": feature,
                    "Model": model,
                    "Best_Triplet_Acc": best_acc,
                    "Final_Loss": final_loss,
                    "Avg_Pos_Dist": avg_pos, # Lower is better
                    "Avg_Neg_Dist": avg_neg  # Higher is better
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(BASE_DIR, "results", "Siamese_Comparison_Table.csv")
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")
        print(df)

# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":
    for feature, model_name in CHAMPIONS.items():
        train_siamese(feature, model_name)
    
    generate_summary()