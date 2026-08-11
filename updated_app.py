import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import json
import uuid
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
from pillow_heif import register_heif_opener
from scipy.stats import norm

register_heif_opener()

# Force PyTorch to use less memory
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ==========================================
# 1. CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2"

app = Flask(__name__, template_folder=CURRENT_DIR, static_folder=os.path.join(PROJECT_ROOT, 'static'))

DB_DIR = os.path.join(PROJECT_ROOT, "website_db")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
TRADEMARK_FOLDER = os.path.join(PROJECT_ROOT, "static", "trademarks")
GLOBAL_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "global_data")
LOCAL_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "local_data")

for d in [UPLOAD_FOLDER, TRADEMARK_FOLDER, DB_DIR]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STYLE_DATABASE = {
    # --- Surrealism & Fantasy ---
    "Zdzislaw Beksinski": "Dystopian Surrealism",
    "Salvador Dali": "Surrealism",
    "Frida Kahlo": "Naïve Art / Symbolism",

    # --- Renaissance ---
    "Sandro Botticelli": "Early Renaissance",
    "Leonardo da Vinci": "High Renaissance",
    "Michelangelo": "High Renaissance",

    # --- Baroque & Dutch Golden Age ---
    "Caravaggio": "Baroque (Chiaroscuro)",
    "Rembrandt": "Baroque Realism",
    "Johannes Vermeer": "Dutch Golden Age",

    # --- Impressionism & Post-Impressionism ---
    "Claude Monet": "Impressionism",
    "Pierre Auguste Renoir": "Impressionism",
    "Vincent Van Gogh": "Post-Impressionism",

    # --- Modernism & Expressionism ---
    "Edvard Munch": "Expressionism",
    "Gustave Dore": "Romanticism",
    "Francisco Goya": "Romanticism",
    "Henri Matisse": "Fauvism",
    "Pablo Picasso": "Cubism",
    "Georgia Okeefe": "American Modernism",

    # --- Abstract & Pop ---
    "Jackson Pollock": "Abstract Expressionism",
    "Andy Warhol": "Pop Art"
}

# --- CHECKPOINTS ---
GLOBAL_CKPTS = {
    'color': os.path.join(PROJECT_ROOT, r"checkpoints\comparison\efficientnet_b3_color_fold4_best.pth"),
    'brushstroke': os.path.join(PROJECT_ROOT, r"checkpoints\comparison\efficientnet_b3_brushstroke_fold1_best.pth"),
    'texture': os.path.join(PROJECT_ROOT, r"checkpoints\comparison\efficientnet_b3_texture_fold1_best.pth")
}
LOCAL_CKPTS = {
    'color': os.path.join(PROJECT_ROOT, r"checkpoints\local_comparison\local_efficientnet_b3_color_fold1_best.pth"),
    'brushstroke': os.path.join(PROJECT_ROOT, r"checkpoints\local_comparison\local_efficientnet_b3_brushstroke_fold1_best.pth"),
    'texture': os.path.join(PROJECT_ROOT, r"checkpoints\local_comparison\local_efficientnet_b3_texture_fold1_best.pth")
}
SIAMESE_CKPTS = {
    'color': os.path.join(PROJECT_ROOT, r"checkpoints\siamese\siamese_color_best.pth"),
    'texture': os.path.join(PROJECT_ROOT, r"checkpoints\siamese\siamese_texture_best.pth"),
    'brushstroke': os.path.join(PROJECT_ROOT, r"checkpoints\siamese\siamese_brushstroke_best.pth")
}

# ==========================================
# 2. MODEL ENGINE (LAZY LOADING)
# ==========================================
def get_classifier_arch(arch, num_classes):
    """ Creates a FRESH classifier model instance """
    if arch == 'resnet50':
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == 'vgg16':
        m = models.vgg16(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif arch == 'alexnet':
        m = models.alexnet(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    return m

class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base = base_model
        if hasattr(self.base, 'fc'):
            num_ftrs = self.base.fc.in_features
            self.base.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 128))
        elif hasattr(self.base, 'classifier'):
            idx = len(self.base.classifier) - 1
            if isinstance(self.base.classifier[idx], nn.Linear):
                num_ftrs = self.base.classifier[idx].in_features
                self.base.classifier[idx] = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Linear(256, 128))
    def forward(self, x): return self.base(x)

def load_checkpoint(model, path):
    if not os.path.exists(path): return None
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        return model.to(device).eval()
    except: return None

def get_siamese_base(arch):
    """ FACTORY: Creates FRESH base model """
    if arch == 'resnet50': m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 20)
    elif arch == 'efficientnet_b3': m = models.efficientnet_b3(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 20)
    return m

def get_model(model_type, feature, scope=None):
    """
    Loads a model on-demand and returns it.
    Args:
        model_type: 'classifier' or 'siamese'
        feature: 'color', 'texture', or 'brushstroke'
        scope: 'global' or 'local' (only for classifiers)
    """
    if model_type == 'classifier':
        if scope == 'global':
            ckpt_path = GLOBAL_CKPTS[feature]
            num_classes = 50  # Updated: 50 global artists
            arch = 'efficientnet_b3'  # All features use EfficientNetB3
        else:  # local
            ckpt_path = LOCAL_CKPTS[feature]
            num_classes = 61  # Updated: 61 local students
            arch = 'efficientnet_b3'  # All features use EfficientNetB3
        
        model = get_classifier_arch(arch, num_classes)
        return load_checkpoint(model, ckpt_path)
    
    elif model_type == 'siamese':
        ckpt_path = SIAMESE_CKPTS[feature]
        arch = 'efficientnet_b3'  # All Siamese models use EfficientNetB3
        base = get_siamese_base(arch)
        model = SiameseNetwork(base)
        return load_checkpoint(model, ckpt_path)
    
    return None

def clear_gpu_cache():
    """Clears GPU memory after processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Model paths configured. Models will load on-demand to save memory.")
TRAINED_G = 50
TRAINED_L = 61

GLOBAL_CLASSES = sorted([d for d in os.listdir(GLOBAL_IMG_DIR) if os.path.isdir(os.path.join(GLOBAL_IMG_DIR, d))])[:TRAINED_G]
LOCAL_CLASSES = sorted([d for d in os.listdir(LOCAL_IMG_DIR) if os.path.isdir(os.path.join(LOCAL_IMG_DIR, d))])[:TRAINED_L]

# ==========================================
# 3. ANALYSIS LOGIC
# ==========================================
def preprocess_image(image_path, feature):
    try:
        stream = open(image_path, "rb"); bytes = bytearray(stream.read()); numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR); stream.close()
        if img is None: return None
        processed = img.copy()
        if feature == 'color':
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            processed = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
        elif feature == 'brushstroke':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            v = np.median(blurred); edges = cv2.Canny(blurred, int(max(0, 0.67*v)), int(min(255, 1.33*v)))
            dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            processed = cv2.cvtColor(cv2.bitwise_not(dilated), cv2.COLOR_GRAY2BGR)
        elif feature == 'texture':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); lbp = local_binary_pattern(gray, 24, 3, method='uniform')
            lbp_norm = (lbp * 255 / 26).astype(np.uint8); processed = cv2.cvtColor(lbp_norm, cv2.COLOR_GRAY2BGR)
        pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return tf(pil).unsqueeze(0).to(device)
    except: return None

def predict_with_classifiers(image_paths, scope, exclude_name=None):
    classes = GLOBAL_CLASSES if scope == 'global' else LOCAL_CLASSES
    votes = {name: 0.0 for name in classes}
    
    for p in image_paths:
        for feat in ['color', 'texture', 'brushstroke']:
            # Load model on-demand
            model = get_model('classifier', feat, scope)
            if model is None: continue
            
            tensor = preprocess_image(p, feat)
            if tensor is not None:
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    score, idx = torch.max(probs, 1)
                    if idx.item() < len(classes): 
                        votes[classes[idx.item()]] += score.item()
            
            # Clear model from memory
            del model
            clear_gpu_cache()
    
    if exclude_name and exclude_name in votes: del votes[exclude_name]
    
    sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    total = sum(votes.values())
    results = []
    if total > 0:
        for name, score in sorted_votes[:3]:
            conf = round((score / total) * 100, 2)
            results.append({"name": name, "score": conf})
    return results

def get_siamese_centroid(image_paths):
    centroid = {}
    for feat in ['color', 'texture', 'brushstroke']:
        # Load model on-demand
        model = get_model('siamese', feat)
        if model is None: continue
        
        embs = []
        for p in image_paths:
            tensor = preprocess_image(p, feat)
            if tensor is not None:
                with torch.no_grad(): 
                    embs.append(model(tensor).cpu().numpy().flatten())
        
        # Clear model from memory
        del model
        clear_gpu_cache()
        
        if embs:
            mean = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean)
            centroid[feat] = mean / norm if norm > 0 else mean
    
    return centroid

def calculate_originality(query_centroid, current_artist_name=None):
    """
    Z-Score Normalized Style Distinctiveness
    
    Measures how statistically different an artist is from the cohort mean.
    Uses standard deviation to account for dataset homogeneity.
    
    Research Basis:
    - Normalizes for cohort characteristics (educational context)
    - Z-scores provide interpretable statistical distance
    - Accounts for dataset variance (tight vs. diverse cohorts)
    """
    l_dir = os.path.join(DB_DIR, "local")
    if not os.path.exists(l_dir): 
        return {
            "score": 85.0,
            "tier": "BASELINE",
            "interpretation": "No comparison cohort available",
            "cohort_size": 0,
            "z_score": 0.0,
            "cohort_mean": 0.0,
            "cohort_std": 0.0
        }
    
    sims = []
    artist_sims = {}
    
    # Collect similarities across all features
    for feat in query_centroid:
        f_dir = os.path.join(l_dir, feat)
        if os.path.exists(f_dir):
            for npy in os.listdir(f_dir):
                artist = npy.replace(".npy", "")
                if artist == current_artist_name: continue 
                
                npy_path = os.path.join(f_dir, npy)
                if not os.path.exists(npy_path): continue
                
                try:
                    db_vec = np.load(npy_path)
                    sim = cosine_similarity(query_centroid[feat].reshape(1,-1), db_vec.reshape(1,-1))[0][0]
                    sims.append(sim)
                    
                    if artist not in artist_sims:
                        artist_sims[artist] = []
                    artist_sims[artist].append(sim)
                except: continue
    
    cohort_size = len(artist_sims)
    
    if not sims or cohort_size == 0: 
        return {
            "score": 95.0,
            "tier": "UNIQUE",
            "interpretation": "No similar artists in database",
            "cohort_size": 0,
            "z_score": 0.0,
            "cohort_mean": 0.0,
            "cohort_std": 0.0
        }
    
    # ========================================
    # STEP 1: Calculate Query Artist's Max Similarity
    # ========================================
    max_sim = max(sims)  # Highest similarity to ANY artist
    avg_sim = np.mean(sims)
    std_sim = np.std(sims)
    
    # Find closest match
    closest_artist = max(artist_sims.items(), key=lambda x: np.mean(x[1]))
    closest_avg_sim = np.mean(closest_artist[1])
    
    # ========================================
    # STEP 2: Collect ALL Max Similarities Across Cohort
    # ========================================
    # This creates the reference distribution
    all_max_sims = []
    for artist in artist_sims.keys():
        artist_max_sim = max(artist_sims[artist])
        all_max_sims.append(artist_max_sim)
    
    # Add current artist's max similarity
    all_max_sims.append(max_sim)
    
    # ========================================
    # STEP 3: Z-Score Normalization
    # ========================================
    cohort_mean = np.mean(all_max_sims)
    cohort_std = np.std(all_max_sims)
    
    # Calculate z-score (how many std devs from mean)
    z_score = (max_sim - cohort_mean) / cohort_std if cohort_std > 0 else 0
    
    # ========================================
    # STEP 4: Convert Z-Score to Originality Scale (0-100)
    # ========================================
    # ADJUSTED MAPPING (More Encouraging):
    # z = -3.0 (very distinctive) → 95% (capped at 99%)
    # z = -2.0 (distinctive)      → 80%
    # z = -1.0 (above average)    → 65%
    # z =  0.0 (average)          → 50%
    # z = +1.0 (below average)    → 35%
    # z = +2.0 (similar)          → 20%
    # z = +3.0 (very similar)     → 5% (capped at 20%)
    
    originality = 50 - (z_score * 15)  # Reduced from 20 to 15
    originality = max(20.0, min(99.0, originality))  # Raised floor from 10% to 20%
    
    # ========================================
    # STEP 5: Tier Classification (Based on Z-Scores)
    # ========================================
    if z_score <= -2.0:
        tier = "HIGHLY DISTINCTIVE"
        tier_desc = f"Exceptionally unique style (more distinct than ~97.7% of cohort)"
        interpretation = f"Remarkably original aesthetic - strong personal voice. Z-score: {z_score:.2f}"

    elif z_score <= -1.0:
        tier = "DISTINCTIVE"
        tier_desc = f"Above-average originality (more distinct than ~84.1% of cohort)"
        interpretation = f"Clear stylistic identity emerging. Z-score: {z_score:.2f}"

    elif z_score <= 0.0:
        tier = "DEVELOPING"
        tier_desc = "Establishing personal style (more distinct than ~50% of cohort)"
        interpretation = f"Building unique visual language. Z-score: {z_score:.2f}"

    elif z_score <= 1.0:
        tier = "COHESIVE"
        tier_desc = "Strong alignment with contemporary trends (more distinct than ~16-50% of cohort)"
        interpretation = f"Well-integrated with current artistic dialogue. Z-score: {z_score:.2f}"

    elif z_score <= 2.0:
        tier = "FOUNDATIONAL"
        tier_desc = f"Building on established influences (more distinct than ~2.3-16% of cohort)"
        interpretation = f"Absorbing inspiration from {closest_artist[0]} - natural learning stage. Z-score: {z_score:.2f}"

    else:
        tier = "EXPLORATORY"
        tier_desc = f"Early development phase (learning from established styles)"
        interpretation = f"Drawing from {closest_artist[0]}'s approach - continue experimenting. Z-score: {z_score:.2f}"
    # ========================================
    # DEBUG LOGGING
    # ========================================
    percentile = norm.cdf(-z_score) * 100 # Negative z-score = higher distinctiveness

    print(f"\n🎯 Z-Score Normalized Distinctiveness for {current_artist_name}:")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   📊 COHORT STATISTICS:")
    print(f"      Total Artists: {cohort_size}")
    print(f"      Mean Max Similarity: {cohort_mean:.4f} ({cohort_mean*100:.2f}%)")
    print(f"      Std Deviation: {cohort_std:.4f}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   🎨 YOUR METRICS:")
    print(f"      Your Max Sim: {max_sim:.4f} ({max_sim*100:.2f}%)")
    print(f"      Your Avg Sim: {avg_sim:.4f} ({avg_sim*100:.2f}%)")
    print(f"      Closest Match: {closest_artist[0]} ({closest_avg_sim*100:.2f}%)")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   📈 Z-SCORE ANALYSIS:")
    print(f"      Z-Score: {z_score:.3f}")
    print(f"      Interpretation: {interpretation}")
    print(f"      Percentile: ~{percentile:.1f}% (more distinctive than {percentile:.1f}% of cohort)")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   ✅ FINAL RESULTS:")
    print(f"      Originality Score: {originality:.2f}%")
    print(f"      Tier: {tier} ({tier_desc})")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # ========================================
    # RETURN COMPREHENSIVE METADATA
    # ========================================
    return {
        "score": originality,
        "tier": tier,
        "tier_description": tier_desc,
        "interpretation": interpretation,
        "cohort_size": cohort_size,
        "z_score": round(z_score, 3),
        "cohort_mean": round(cohort_mean, 4),
        "cohort_std": round(cohort_std, 4),
        "max_similarity": round(max_sim, 4),
        "closest_match": closest_artist[0],
        "closest_match_score": round(closest_avg_sim * 100, 2)
    }

'''
def generate_trademark(image_paths, artist_name):
    images = []
    for p in image_paths:
        try:
            stream = open(p, "rb"); bytes = bytearray(stream.read()); numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR); stream.close()
            if img is not None: images.append(cv2.resize(img, (300, 300)).astype(np.float32))
        except: continue
    if not images: return None, None
    mean_img = np.mean(images, axis=0).astype(np.uint8)
    filename = f"{artist_name}_trademark.jpg"
    cv2.imwrite(os.path.join(TRADEMARK_FOLDER, filename), mean_img)
    return filename, os.path.join(TRADEMARK_FOLDER, filename)
'''


def extract_signature_region(img):
    """
    Extracts the most representative 300x300 region from an artwork
    Prioritizes: 1) Curved patterns/swirls, 2) Dominant colors, 3) Texture
    """
    h, w = img.shape[:2]
    
    # If image too small, return center crop
    if h < 300 or w < 300:
        return cv2.resize(img, (300, 300))
    
    # ========================================
    # Multi-Feature Saliency Analysis
    # ========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. EDGE DENSITY & CURVATURE (captures swirls/brushstrokes)
    edges = cv2.Canny(gray, 50, 150)
    
    # 2. DETECT CURVED PATTERNS (Hough Circles for swirls)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    
    # 3. TEXTURE VARIANCE
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    
    # 4. COLOR SATURATION (HSV for dominant colors)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ========================================
    # Sliding Window to Find Best Region
    # ========================================
    best_score = -1
    best_crop = None
    step = 40  # Reduced step for finer search
    
    for y in range(0, h - 300, step):
        for x in range(0, w - 300, step):
            # Extract 300x300 window
            window_edges = edges[y:y+300, x:x+300]
            window_lbp = lbp[y:y+300, x:x+300]
            window_hsv = hsv[y:y+300, x:x+300]
            
            # Score 1: Edge density + curvature bonus
            edge_score = np.sum(window_edges > 0) / (300 * 300)
            
            # Check if circles detected in this region
            curvature_bonus = 0
            if circles is not None:
                for circle in circles[0]:
                    cx, cy, r = circle
                    if x <= cx < x+300 and y <= cy < y+300:
                        curvature_bonus += 0.2  # Bonus for curved patterns
            
            # Score 2: Color saturation (dominant colors)
            saturation = window_hsv[:,:,1]
            saturation_score = np.mean(saturation) / 255.0
            
            # Score 3: Texture variance
            texture_score = np.std(window_lbp) / 10.0
            
            # Combined score (REWEIGHTED for pattern detection)
            combined_score = (
                (edge_score * 0.3) + 
                (curvature_bonus * 0.3) +  # NEW: Prioritize curves
                (saturation_score * 0.25) +  # Dominant colors
                (texture_score * 0.15)
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_crop = img[y:y+300, x:x+300]
    
    # Fallback to center crop if no good region found
    if best_crop is None:
        cy, cx = h // 2, w // 2
        best_crop = img[cy-150:cy+150, cx-150:cx+150]
    
    return best_crop

def generate_trademark(image_paths, artist_name):
    """
    Creates a 3x3 grid mosaic of SIGNATURE STYLE REGIONS
    Each tile shows a cropped section that captures recurring patterns:
    - Color palette concentrations
    - Brushstroke characteristics  
    - Textural motifs
    """
    if not image_paths:
        return None, None
    
    print(f"\n🎨 Generating Trademark Mosaic for {artist_name}...")
    
    # ========================================
    # STEP 1: Select Representative Samples
    # ========================================
    # Get embeddings for all artworks
    embeddings = []
    for path in image_paths:
        emb_dict = {}
        
        for feat in ['color', 'texture', 'brushstroke']:
            model = get_model('siamese', feat)
            if model is None: continue
            
            tensor = preprocess_image(path, feat)
            if tensor is not None:
                with torch.no_grad():
                    emb = model(tensor).cpu().numpy().flatten()
                    emb_dict[feat] = emb
            
            del model
            clear_gpu_cache()
        
        if len(emb_dict) == 3:
            combined = np.concatenate([emb_dict[f] for f in ['color', 'texture', 'brushstroke']])
            embeddings.append((path, combined))
    
    if not embeddings:
        return None, None
    
    # ========================================
    # STEP 2: Select 9 DIVERSE Representative Works
    # ========================================
    all_embs = np.array([e[1] for e in embeddings])
    centroid = np.mean(all_embs, axis=0)
    
    # Find closest to centroid (most representative)
    distances = [np.linalg.norm(emb - centroid) for _, emb in embeddings]
    closest_idx = np.argmin(distances)
    
    # Select 9 samples: 1 center + 8 diverse
    selected_indices = [closest_idx]  # Start with most representative
    
    # Add diverse samples using k-means++ style selection
    from sklearn.cluster import KMeans
    if len(embeddings) >= 9:
        kmeans = KMeans(n_clusters=min(9, len(embeddings)), random_state=42, n_init=10)
        kmeans.fit(all_embs)
        
        # Get one sample from each cluster
        for cluster_id in range(min(9, kmeans.n_clusters)):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Get closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                cluster_distances = [np.linalg.norm(all_embs[i] - cluster_center) for i in cluster_indices]
                closest_in_cluster = cluster_indices[np.argmin(cluster_distances)]
                if closest_in_cluster not in selected_indices:
                    selected_indices.append(closest_in_cluster)
    else:
        # If < 9 images, use all
        selected_indices = list(range(len(embeddings)))
    
    # Limit to 9
    selected_indices = selected_indices[:9]
    
    # ========================================
    # STEP 3: Extract SIGNATURE REGIONS (NEW!)
    # ========================================
    tiles = []
    for idx in selected_indices:
        img_path = embeddings[idx][0]
        img = cv2.imread(img_path)
        if img is not None:
            # Extract most representative 300x300 region
            signature_region = extract_signature_region(img)
            # Resize to 200x200 for grid
            tiles.append(cv2.resize(signature_region, (200, 200)))
    
    # ========================================
    # FILL TO 9 TILES (No Black Padding!)
    # ========================================
    # If fewer than 9 artworks, intelligently reuse regions
    if len(tiles) < 9 and len(tiles) > 0:
        # Strategy: Extract multiple regions from available artworks
        needed = 9 - len(tiles)
        
        # Cycle through artworks to extract additional diverse regions
        for i in range(needed):
            # Round-robin through available artworks
            idx = selected_indices[i % len(selected_indices)]
            img_path = embeddings[idx][0]
            img = cv2.imread(img_path)
            
            if img is not None:
                h, w = img.shape[:2]
                
                # Extract from different quadrants to ensure diversity
                quadrant = i % 4
                if h >= 300 and w >= 300:
                    if quadrant == 0:  # Top-left
                        region = img[0:300, 0:300]
                    elif quadrant == 1:  # Top-right
                        region = img[0:300, max(0, w-300):w]
                    elif quadrant == 2:  # Bottom-left
                        region = img[max(0, h-300):h, 0:300]
                    else:  # Bottom-right
                        region = img[max(0, h-300):h, max(0, w-300):w]
                    
                    tiles.append(cv2.resize(region, (200, 200)))
                else:
                    # If image too small, extract signature region again
                    tiles.append(cv2.resize(extract_signature_region(img), (200, 200)))
    
    # ========================================
    # STEP 4: Create Grid Layout
    # ========================================
    rows = []
    for i in range(0, 9, 3):
        row = np.hstack(tiles[i:i+3])
        rows.append(row)
    
    grid = np.vstack(rows)
    
    # ========================================
    # STEP 5: Save Grid
    # ========================================
    filename = f"{artist_name}_trademark.jpg"
    dest_path = os.path.join(TRADEMARK_FOLDER, filename)
    cv2.imwrite(dest_path, grid)
    
    print(f"   ✅ Created 3x3 signature region mosaic")
    print(f"   📊 Extracted distinctive regions from {len(selected_indices)} works")
    print(f"   🎯 Portfolio size: {len(image_paths)} works")
    
    return filename, dest_path

# REPLACE generate_detailed_description function (around line 565)

def generate_detailed_description(image_paths, global_matches):
    """
    Enhanced Trademark Analysis:
    1. Color palette (existing)
    2. Recurring visual motifs (NEW)
    3. Compositional patterns (NEW)
    4. Style differentiators (NEW)
    """
    try:
        # Load the trademark image (already generated)
        trademark_path = None
        for p in image_paths:
            artist_name = os.path.basename(os.path.dirname(p))
            potential_tm = os.path.join(TRADEMARK_FOLDER, f"{artist_name}_trademark.jpg")
            if os.path.exists(potential_tm):
                trademark_path = potential_tm
                break
        
        if not trademark_path:
            return "Trademark not generated yet."
        
        # Read trademark image
        img = cv2.imread(trademark_path)
        if img is None:
            return "Unable to analyze trademark."
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ========================================
        # 1. DOMINANT COLOR ANALYSIS (Existing)
        # ========================================
        pixels = img_rgb.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        sorted_indices = np.argsort(-counts)
        dominant_colors = colors[sorted_indices]
        
        color_names = [get_color_name(color) for color in dominant_colors]
        color_names_unique = []
        for name in color_names:
            if name not in color_names_unique:
                color_names_unique.append(name)
        
        if len(color_names_unique) == 3:
            color_desc = f"{color_names_unique[0]}, {color_names_unique[1]}, and {color_names_unique[2]}"
        elif len(color_names_unique) == 2:
            color_desc = f"{color_names_unique[0]} and {color_names_unique[1]}"
        else:
            color_desc = f"{color_names_unique[0]}"
        
        # ========================================
        # 2. EDGE/CONTOUR ANALYSIS (Motif Detection)
        # ========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contour characteristics
        num_major_shapes = len([c for c in contours if cv2.contourArea(c) > 500])
        total_edge_pixels = np.sum(edges > 0)
        edge_density = total_edge_pixels / (edges.shape[0] * edges.shape[1])
        
        # Circularity analysis (detect curves vs. angular shapes)
        circular_count = 0
        angular_count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
                    if circularity > 0.7:
                        circular_count += 1
                    else:
                        angular_count += 1
        
        # Generate motif description
        motif_traits = []
        
        if circular_count > angular_count:
            motif_traits.append("curved, organic forms")
        elif angular_count > circular_count:
            motif_traits.append("angular, geometric shapes")
        else:
            motif_traits.append("mixed curvilinear and angular elements")
        
        if num_major_shapes > 15:
            motif_traits.append("dense composition with multiple focal points")
        elif num_major_shapes > 5:
            motif_traits.append("moderate compositional complexity")
        else:
            motif_traits.append("minimalist focal arrangement")
        
        # ========================================
        # 3. TEXTURE/BRUSHWORK ANALYSIS
        # ========================================
        # Local Binary Pattern for texture
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=10, range=(0, 10))[0]
        texture_variance = np.var(lbp_hist)
        
        # Edge directional analysis (detect brushstroke patterns)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate dominant edge direction
        angles = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
        angle_hist, _ = np.histogram(angles.ravel(), bins=8, range=(-180, 180))
        dominant_direction_idx = np.argmax(angle_hist)
        
        # Brushwork description
        brushwork_traits = []
        
        if texture_variance > 50:
            brushwork_traits.append("highly textured surface with visible mark-making")
        elif texture_variance > 20:
            brushwork_traits.append("moderate surface texture")
        else:
            brushwork_traits.append("smooth, refined surface treatment")
        
        # Directional brushwork
        if angle_hist[dominant_direction_idx] > np.sum(angle_hist) * 0.4:
            if -45 <= (dominant_direction_idx * 45 - 180) <= 45:
                brushwork_traits.append("predominantly horizontal strokes")
            elif 45 <= abs(dominant_direction_idx * 45 - 180) <= 135:
                brushwork_traits.append("predominantly vertical strokes")
            else:
                brushwork_traits.append("predominantly diagonal strokes")
        else:
            brushwork_traits.append("multidirectional brushwork")
        
        # ========================================
        # 4. TONAL CONTRAST (Existing)
        # ========================================
        avg_brightness = np.mean(gray)
        contrast_std = np.std(gray)
        
        tonal_parts = []
        if avg_brightness < 85:
            tonal_parts.append("low-key lighting")
        elif avg_brightness > 170:
            tonal_parts.append("high-key illumination")
        else:
            tonal_parts.append("balanced mid-tone range")
        
        if contrast_std > 60:
            tonal_parts.append("dramatic chiaroscuro effects")
        elif contrast_std > 35:
            tonal_parts.append("clear tonal separation")
        else:
            tonal_parts.append("subtle tonal transitions")
        
        # ========================================
        # 5. COMPARATIVE CONTEXT (NEW)
        # ========================================
        style_influence = ""
        if global_matches:
            top_match = global_matches[0]['name']
            if top_match in STYLE_DATABASE:
                movement = STYLE_DATABASE[top_match]
                style_influence = f" echoing {movement} characteristics"
        
        # ========================================
        # 6. FINAL CONSOLIDATED DESCRIPTION
        # ========================================
        desc = (
            f"Signature palette of {color_desc} tones. "
            f"Features {motif_traits[0]} with {motif_traits[1]}. "
            f"Exhibits {brushwork_traits[0]} with {brushwork_traits[1]}. "
            f"Demonstrates {tonal_parts[0]} and {tonal_parts[1]}{style_influence}."
        )
        
        # ========================================
        # DEBUG OUTPUT
        # ========================================
        print(f"\n🎨 Enhanced Trademark Analysis:")
        print(f"   Primary Colors: {color_names_unique}")
        print(f"   Shapes Detected: {num_major_shapes} | Circular: {circular_count} | Angular: {angular_count}")
        print(f"   Texture Variance: {texture_variance:.2f}")
        print(f"   Brightness: {avg_brightness:.1f} | Contrast: {contrast_std:.1f}")
        print(f"   Edge Density: {edge_density:.4f}")
        
        return desc

    except Exception as e:
        print(f"Description generation error: {e}")
        return "Analysis unavailable."

def get_color_name(rgb):
    """Maps RGB to descriptive color names"""
    r, g, b = rgb
    
    # Grayscale check
    if max(r, g, b) - min(r, g, b) < 30:
        if r < 50: return "Deep Black"
        elif r < 100: return "Charcoal Gray"
        elif r < 150: return "Neutral Gray"
        elif r < 200: return "Light Gray"
        else: return "Off-White"
    
    # Hue-based classification
    if r > g and r > b:
        if g > b: return "Warm Orange-Red" if g > 100 else "Crimson Red"
        else: return "Magenta" if b > 100 else "Scarlet Red"
    
    elif g > r and g > b:
        if r > b: return "Yellow-Green" if r > 120 else "Lime Green"
        else: return "Emerald Green" if b < 100 else "Teal Green"
    
    elif b > r and b > g:
        if r > g: return "Violet Blue" if r > 100 else "Deep Blue"
        else: return "Cyan Blue" if g > 100 else "Navy Blue"
    
    return "Complex Mixed Tone"

# ==========================================
# 4. UNIFIED PROCESSING
# ==========================================

def find_closest_matches(query_centroid, scope='global', exclude_name=None):
    """ Finds top 3 matches using Cosine Similarity on .npy files """
    target_dir = os.path.join(DB_DIR, scope)
    scores = {} # {artist: [sim_color, sim_tex, sim_brush]}
    
    # 1. Scan Database
    for feat in query_centroid:
        f_dir = os.path.join(target_dir, feat)
        if os.path.exists(f_dir):
            for npy in os.listdir(f_dir):
                artist = npy.replace(".npy", "")
                if artist == exclude_name: continue # Skip Self
                
                try:
                    db_vec = np.load(os.path.join(f_dir, npy))
                    sim = cosine_similarity(query_centroid[feat].reshape(1,-1), db_vec.reshape(1,-1))[0][0]
                    if artist not in scores: scores[artist] = []
                    scores[artist].append(sim)
                except: continue

    # 2. Average Scores
    results = []
    for artist, sim_list in scores.items():
        if len(sim_list) > 0:
            avg_sim = float(np.mean(sim_list))
            results.append({"name": artist, "score": round(avg_sim * 100, 2)})
            
    # 3. Sort by Similarity
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:3]

def process_and_update_artist(name):
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, name)): artist_dir = os.path.join(LOCAL_IMG_DIR, name)
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, name)): artist_dir = os.path.join(GLOBAL_IMG_DIR, name)
    else: return False
    
    images = [os.path.join(artist_dir, f) for f in os.listdir(artist_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    if not images: return False
    
    # 1. Get Centroid
    query = get_siamese_centroid(images)
    
    # 2. Find Matches (Siamese Similarity)
    global_res = find_closest_matches(query, 'global', exclude_name=name)
    local_res = find_closest_matches(query, 'local', exclude_name=name)
    
    # 3. Originality (NOW RETURNS DICT)
    orig_result = calculate_originality(query, current_artist_name=name)
    
    # Handle both old (float) and new (dict) return types for backward compatibility
    if isinstance(orig_result, dict):
        orig_score = orig_result["score"]
        cohort_size = orig_result["cohort_size"]
        tier = orig_result.get("tier", "")
        interpretation = orig_result.get("interpretation", "")
    else:
        # Legacy support (if you revert)
        orig_score = orig_result
        cohort_size = len([f for f in os.listdir(os.path.join(DB_DIR, "local", "color")) if f.endswith('.npy')]) - 1
        tier = ""
        interpretation = ""
    
    # 4. Visuals
    tm_file, tm_path = generate_trademark(images, name)
    desc = generate_detailed_description(images, global_res)
    
    # 5. Save JSON (NOW INCLUDES COHORT SIZE)
    def clean(res): 
        return [{"name": i["name"], "score": float(i["score"])} for i in res]
    
    data = {
        "global_matches": clean(global_res),
        "local_matches": clean(local_res),
        "originality": "{:.2f}".format(orig_score),
        "cohort_size": cohort_size,  # ADDED
        "tier": tier,                 # OPTIONAL: for future use
        "interpretation": interpretation,  # OPTIONAL
        "trademark": tm_file,
        "description": desc
    }
    
    with open(os.path.join(artist_dir, "analysis_results.json"), 'w') as f: 
        json.dump(data, f, indent=2)
    
    # 6. Update NPY (Brain)
    if "local_data" in artist_dir:
        for feat, vec in query.items():
            save_dir = os.path.join(DB_DIR, "local", feat); os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{name}.npy"), vec)
    
    return True

# ==========================================
# 5. ROUTES
# ==========================================
@app.route('/')
def index():
    local, global_a = [], []
    valid = ('.jpg','.png','.jpeg')
    
    if os.path.exists(LOCAL_IMG_DIR):
        for name in os.listdir(LOCAL_IMG_DIR):
            path = os.path.join(LOCAL_IMG_DIR, name)
            if os.path.isdir(path):
                thumb = None
                for f in os.listdir(path):
                    if f.lower().endswith(valid): thumb = f; break
                local.append({'name': name, 'thumbnail': thumb})
    
    if os.path.exists(GLOBAL_IMG_DIR):
        for name in os.listdir(GLOBAL_IMG_DIR):
            path = os.path.join(GLOBAL_IMG_DIR, name)
            if os.path.isdir(path):
                thumb = None
                for f in os.listdir(path):
                    if f.lower().endswith(valid): thumb = f; break
                global_a.append({'name': name, 'thumbnail': thumb})

    return render_template('index.html', local_artists=local, global_artists=global_a)

@app.route('/artist/<name>')
def artist_profile(name):
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, name)): path=os.path.join(LOCAL_IMG_DIR, name); atype="Local Student"
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, name)): path=os.path.join(GLOBAL_IMG_DIR, name); atype="Global Master"
    else: return "Not Found", 404
    
    images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    analysis = None
    
    if atype == "Local Student":
        json_path = os.path.join(path, "analysis_results.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f: analysis = json.load(f)
            except json.JSONDecodeError: analysis = None
            
    return render_template('artist.html', name=name, type=atype, images=images, analysis=analysis)

@app.route('/analyze_profile', methods=['POST'])
def analyze_profile():
    name = request.form.get('name')
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, name)): artist_dir = os.path.join(LOCAL_IMG_DIR, name)
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, name)): 
        return "Action Forbidden: Cannot re-analyze Global Masters.", 403
    else: return "Error", 404
    
    if process_and_update_artist(name):
        return redirect(url_for('artist_profile', name=name))
    return "Error Processing", 500

@app.route('/add_artwork', methods=['POST'])
def add_artwork():
    name = request.form.get('name')
    files = request.files.getlist('files[]')
    
    target_dir = os.path.join(LOCAL_IMG_DIR, name)
    os.makedirs(target_dir, exist_ok=True)
    
    for f in files:
        if f.filename: f.save(os.path.join(target_dir, f.filename))
    
    # Auto-update on upload
    process_and_update_artist(name)
    return redirect(url_for('artist_profile', name=name))

@app.route('/delete_portfolio', methods=['POST'])
def delete_portfolio():
    data = request.get_json()
    name = data.get('name')
    
    target_dir = os.path.join(LOCAL_IMG_DIR, name)
    if not os.path.exists(target_dir):
        return jsonify({"error": "Portfolio not found"}), 404
    
    # Delete folder
    shutil.rmtree(target_dir)
    
    # Delete NPY embeddings
    for feat in ['color', 'texture', 'brushstroke']:
        npy_path = os.path.join(DB_DIR, "local", feat, f"{name}.npy")
        if os.path.exists(npy_path):
            os.remove(npy_path)
    
    # Delete trademark
    tm_path = os.path.join(TRADEMARK_FOLDER, f"{name}_trademark.jpg")
    if os.path.exists(tm_path):
        os.remove(tm_path)
    
    return jsonify({"success": True})

@app.route('/rename_portfolio', methods=['POST'])
def rename_portfolio():
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    
    old_dir = os.path.join(LOCAL_IMG_DIR, old_name)
    new_dir = os.path.join(LOCAL_IMG_DIR, new_name)
    
    if not os.path.exists(old_dir):
        return jsonify({"error": "Portfolio not found"}), 404
    
    if os.path.exists(new_dir):
        return jsonify({"error": "Portfolio name already exists"}), 400
    
    # Rename folder
    os.rename(old_dir, new_dir)
    
    # Rename NPY embeddings
    for feat in ['color', 'texture', 'brushstroke']:
        old_npy = os.path.join(DB_DIR, "local", feat, f"{old_name}.npy")
        new_npy = os.path.join(DB_DIR, "local", feat, f"{new_name}.npy")
        if os.path.exists(old_npy):
            os.rename(old_npy, new_npy)
    
    # Rename trademark
    old_tm = os.path.join(TRADEMARK_FOLDER, f"{old_name}_trademark.jpg")
    new_tm = os.path.join(TRADEMARK_FOLDER, f"{new_name}_trademark.jpg")
    if os.path.exists(old_tm):
        os.rename(old_tm, new_tm)
    
    return jsonify({"success": True})

@app.route('/rename_artwork', methods=['POST'])
def rename_artwork():
    data = request.get_json()
    artist = data.get('artist')
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    
    # Check both local and global directories
    target_dir = None
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, artist)):
        target_dir = os.path.join(LOCAL_IMG_DIR, artist)
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, artist)):
        target_dir = os.path.join(GLOBAL_IMG_DIR, artist)
    else:
        return jsonify({"error": "Artist not found"}), 404
    
    old_path = os.path.join(target_dir, old_name)
    new_path = os.path.join(target_dir, new_name)
    
    if not os.path.exists(old_path):
        return jsonify({"error": "File not found"}), 404
    
    if os.path.exists(new_path):
        return jsonify({"error": "Filename already exists"}), 400
    
    os.rename(old_path, new_path)
    return jsonify({"success": True})

@app.route('/delete_artwork', methods=['POST'])
def delete_artwork():
    data = request.get_json()
    artist = data.get('artist')
    filename = data.get('filename')
    
    # Check both local and global directories
    target_dir = None
    is_local = False
    
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, artist)):
        target_dir = os.path.join(LOCAL_IMG_DIR, artist)
        is_local = True
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, artist)):
        target_dir = os.path.join(GLOBAL_IMG_DIR, artist)
    else:
        return jsonify({"error": "Artist not found"}), 404
    
    file_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    os.remove(file_path)
    
    # Re-analyze profile after deletion (ONLY for local students)
    if is_local:
        remaining = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if remaining:
            process_and_update_artist(artist)
    
    return jsonify({"success": True})

@app.route('/get_image/<name>/<filename>')
def get_image(name, filename):
    if os.path.exists(os.path.join(LOCAL_IMG_DIR, name, filename)): return send_from_directory(os.path.join(LOCAL_IMG_DIR, name), filename)
    elif os.path.exists(os.path.join(GLOBAL_IMG_DIR, name, filename)): return send_from_directory(os.path.join(GLOBAL_IMG_DIR, name), filename)
    return "", 404

@app.route('/get_trademark/<filename>')
def get_trademark(filename): return send_from_directory(TRADEMARK_FOLDER, filename)

if __name__ == '__main__': app.run(debug=True)