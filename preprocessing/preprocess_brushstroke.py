"""
Unified Brushstroke Preprocessing Script (Canny Edge Detection)
Applies consistent Canny edge detection with dilation to both Global and Local datasets.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# MODE: "Global" or "Local"
MODE = "Global"  # Testing Dynamic Canny for final validation

# Configuration
BASE_DIR = r"C:\Users\PC\OneDrive\Documents\ArtModelThesis2\data"

if MODE == "Global":
    INPUT_DIR = os.path.join(BASE_DIR, "aug_global_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "global_data_brushstroke")
elif MODE == "Local":
    INPUT_DIR = os.path.join(BASE_DIR, "aug_local_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed", "local_data_brushstroke")
else:
    raise ValueError("MODE must be 'Global' or 'Local'")

# Edge detection parameters (STANDARDIZED for both Global and Local)
USE_DYNAMIC_CANNY = False  # Fixed thresholds for consistent comparison
LOWER_THRESHOLD = 50
UPPER_THRESHOLD = 150

DILATION_KERNEL_SIZE = 3
DILATION_ITERATIONS = 1

canny_desc = "Dynamic thresholds" if USE_DYNAMIC_CANNY else f"Fixed thresholds ({LOWER_THRESHOLD}, {UPPER_THRESHOLD})"
print(f"🖌️  Brushstroke Preprocessing ({MODE} Dataset)")
print(f"📂 Input:  {INPUT_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")
print(f"⚙️  Canny: {canny_desc}, Dilation kernel={DILATION_KERNEL_SIZE}x{DILATION_KERNEL_SIZE}, iterations={DILATION_ITERATIONS}")
print("=" * 80)

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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny thresholds
            if USE_DYNAMIC_CANNY:
                median_intensity = np.median(blurred)
                lower = int(max(0, 0.7 * median_intensity))
                upper = int(min(255, 1.3 * median_intensity))
            else:
                lower = LOWER_THRESHOLD
                upper = UPPER_THRESHOLD
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, lower, upper)
            
            # Apply dilation to thicken edges
            kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
            
            # Save image (Unicode-safe)
            output_path = os.path.join(artist_output_path, img_file)
            ext = os.path.splitext(img_file)[1]
            is_success, buffer = cv2.imencode(ext, edges_dilated)
            
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
