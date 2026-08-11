import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.manifold import TSNE
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis"
OUTPUT_DIR = os.path.join(BASE_DIR, r"results\tsne_parallel_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = os.path.join(BASE_DIR, r"checkpoints\siamese")
DATA_GLOBAL = os.path.join(BASE_DIR, r"data\global_data")
DATA_LOCAL = os.path.join(BASE_DIR, r"data\local_data")

# Top 10 Artists for visualization
TARGET_GLOBAL = [
    "Claude Monet", "Henri Matisse", "Pablo Picasso", "Pierre Auguste Renoir",
    "Salvador Dali", "Vincent Van Gogh", "Rembrandt", "Gustave Dore",
    "Zdzislaw Beksinski", "Edvard Munch"
]

TARGET_LOCAL = [
    "John Laurence Pretista", "Jeneviv Salcedo", "Marc Renzie Gutierrez",
    "Jovany Bicay", "Almar Terante", "Rhea Larivee Labesores",
    "Kenneth Villegas", "Angel Comparativo", "Daniel Saballa", "Mark Mendoza"
]

MODELS_CONFIG = {
    'color': {'arch': 'resnet50', 'pth': 'siamese_color_best.pth'},
    'texture': {'arch': 'efficientnet_b3', 'pth': 'siamese_texture_best.pth'},
    'brushstroke': {'arch': 'efficientnet_b3', 'pth': 'siamese_brushstroke_best.pth'}
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base = base_model
        if hasattr(self.base, 'fc'):
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 128))
        elif hasattr(self.base, 'classifier'):
            if isinstance(self.base.classifier, nn.Sequential):
                idx = len(self.base.classifier) - 1
                num_ftrs = self.base.classifier[idx].in_features
                self.base.classifier[idx] = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 128))
    
    def forward(self, x):
        return self.base(x)

def get_base_model(arch):
    if arch == 'resnet50':
        return models.resnet50(weights=None)
    elif arch == 'efficientnet_b3':
        return models.efficientnet_b3(weights=None)
    return None

# ==========================================
# 3. FEATURE PREPROCESSING
# ==========================================
def preprocess_image(img_path, feature_type):
    """Apply feature-specific preprocessing"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    if feature_type == 'color':
        # CLAHE normalization
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        processed = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    elif feature_type == 'brushstroke':
        # Canny edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        v = np.median(blurred)
        edges = cv2.Canny(blurred, int(max(0, 0.67 * v)), int(min(255, 1.33 * v)))
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        processed = cv2.cvtColor(cv2.bitwise_not(dilated), cv2.COLOR_GRAY2RGB)
    
    elif feature_type == 'texture':
        # Local Binary Pattern
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        lbp_norm = (lbp * 255 / 26).astype(np.uint8)
        processed = cv2.cvtColor(lbp_norm, cv2.COLOR_GRAY2RGB)
    
    else:
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(processed)
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return tf(pil_img)

# ==========================================
# 4. EXTRACT EMBEDDINGS
# ==========================================
def extract_embeddings(model, data_dir, target_artists, feature_type, max_per_artist=30):
    """Extract embeddings for specified artists"""
    embeddings = []
    labels = []
    
    model.eval()
    
    for artist in target_artists:
        artist_path = os.path.join(data_dir, artist)
        if not os.path.exists(artist_path):
            print(f"   [Warning] Artist not found: {artist}")
            continue
        
        count = 0
        for img_name in os.listdir(artist_path):
            if count >= max_per_artist:
                break
            
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            try:
                img_path = os.path.join(artist_path, img_name)
                tensor = preprocess_image(img_path, feature_type)
                
                if tensor is None:
                    continue
                
                tensor = tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb = model(tensor).cpu().numpy().flatten()
                
                embeddings.append(emb)
                labels.append(artist)
                count += 1
                
            except Exception as e:
                continue
    
    return np.array(embeddings), labels

# ==========================================
# 5. GENERATE PARALLEL t-SNE PLOTS
# ==========================================
def generate_parallel_tsne():
    print("\n" + "="*70)
    print("   GENERATING PARALLEL t-SNE VISUALIZATIONS")
    print("="*70)
    
    # Color palette for consistent visualization
    colors_palette = sns.color_palette("tab10", 10)
    
    for feature, config in MODELS_CONFIG.items():
        print(f"\n📊 Processing {feature.upper()}...")
        
        # Load model
        ckpt_path = os.path.join(CKPT_DIR, config['pth'])
        if not os.path.exists(ckpt_path):
            print(f"   [ERROR] Checkpoint not found: {ckpt_path}")
            continue
        
        try:
            base = get_base_model(config['arch'])
            
            # Mock linear layer to match structure
            if config['arch'] == 'resnet50':
                base.fc = nn.Linear(base.fc.in_features, 20)
            elif config['arch'] == 'efficientnet_b3':
                base.classifier[1] = nn.Linear(base.classifier[1].in_features, 20)
            
            model = SiameseNetwork(base)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state, strict=False)
            model.to(device)
            
            print(f"   ✓ Model loaded: {config['arch']}")
            
        except Exception as e:
            print(f"   [ERROR] Loading model: {e}")
            continue
        
        # Create figure with 2 subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Process Global and Local
        for idx, (scope, data_dir, target_list, ax) in enumerate([
            ('Global', DATA_GLOBAL, TARGET_GLOBAL, axes[0]),
            ('Local', DATA_LOCAL, TARGET_LOCAL, axes[1])
        ]):
            print(f"   → Extracting {scope} embeddings...")
            
            # Extract embeddings
            embeddings, labels = extract_embeddings(model, data_dir, target_list, feature, max_per_artist=30)
            
            if len(embeddings) == 0:
                print(f"   [WARNING] No embeddings extracted for {scope}")
                continue
            
            print(f"   → Running t-SNE for {scope} ({len(embeddings)} samples)...")
            
            # Apply t-SNE
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Plot
            unique_artists = sorted(list(set(labels)))
            
            for i, artist in enumerate(unique_artists):
                mask = np.array([label == artist for label in labels])
                ax.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors_palette[i]],
                    label=artist,
                    s=120,  # Larger dots
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Styling - ENHANCED READABILITY
            title = f"t-SNE Clusters: {feature.capitalize()} ({scope} Top 10)"
            ax.set_title(title, fontsize=24, fontweight='bold', pad=20)  # Increased from 18 to 24
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('t-SNE Dimension 1', fontsize=18)  # Increased from 14 to 18
            ax.set_ylabel('t-SNE Dimension 2', fontsize=18)  # Increased from 14 to 18
            
            # Larger tick labels
            ax.tick_params(axis='both', which='major', labelsize=14)  # NEW: tick label size
            
            # Legend with LARGER font
            ax.legend(
                loc='upper right',
                fontsize=14,  # Increased from 12 to 14
                framealpha=0.95,
                edgecolor='black',
                fancybox=True,
                shadow=True,
                markerscale=1.5  # Increased from 1.3 to 1.5 for larger legend markers
            )
            
            print(f"   ✓ {scope} plot completed")
        
        # Save figure
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"Figure_tSNE_{feature.capitalize()}_Global_Local.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    print("\n" + "="*70)
    print(f"   ALL FIGURES SAVED TO: {OUTPUT_DIR}")
    print("="*70 + "\n")
    
# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    generate_parallel_tsne()