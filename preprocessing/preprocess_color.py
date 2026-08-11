"""
Unified Color Preprocessing Script (CLAHE)
Applies consistent CLAHE preprocessing to both Global and Local datasets.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# MODE: "Global" or "Local"
MODE = "Local"  # Change to "Local" for local dataset preprocessing

# Configuration
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2\data"

if MODE == "Global":
    INPUT_DIR = os.path.join(BASE_DIR, "aug_global_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "global_data_color")
elif MODE == "Local":
    INPUT_DIR = os.path.join(BASE_DIR, "aug_local_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "local_data_color")
else:
    raise ValueError("MODE must be 'Global' or 'Local'")

# CLAHE parameters (consistent for both modes)
CLIP_LIMIT = 3.0
TILE_GRID_SIZE = (8, 8)

print(f"🎨 Color Preprocessing ({MODE} Dataset)")
print(f"📂 Input:  {INPUT_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")
print(f"⚙️  CLAHE: clipLimit={CLIP_LIMIT}, tileGridSize={TILE_GRID_SIZE}")
print("=" * 80)

# Create CLAHE object
clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)

# Get all artist folders
artist_folders = [d for d in os.listdir(INPUT_DIR) 
                  if os.path.isdir(os.path.join(INPUT_DIR, d))]
artist_folders.sort()

total_processed = 0
total_errors = 0

for artist_name in tqdm(artist_folders, desc="Processing Artists"):
    artist_input_path = os.path.join(INPUT_DIR, artist_name)
    artist_output_path = os.path.join(OUTPUT_DIR, artist_name)
    
    # Create output directory
    os.makedirs(artist_output_path, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(artist_input_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        try:
            # Read image (Unicode-safe)
            img_path = os.path.join(artist_input_path, img_file)
            img_array = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"❌ Failed to read: {artist_name}/{img_file}")
                total_errors += 1
                continue
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to BGR
            img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
            
            # Save image (Unicode-safe)
            output_path = os.path.join(artist_output_path, img_file)
            ext = os.path.splitext(img_file)[1]
            is_success, buffer = cv2.imencode(ext, img_clahe)
            
            if is_success:
                buffer.tofile(output_path)
                total_processed += 1
            else:
                print(f"❌ Failed to encode: {artist_name}/{img_file}")
                total_errors += 1
                
        except Exception as e:
            print(f"❌ Error processing {artist_name}/{img_file}: {str(e)}")
            total_errors += 1

print("\n" + "=" * 80)
print(f"✅ Preprocessing Complete!")
print(f"📊 Processed: {total_processed} images")
print(f"❌ Errors: {total_errors}")
print(f"📁 Output: {OUTPUT_DIR}")
print("=" * 80)
