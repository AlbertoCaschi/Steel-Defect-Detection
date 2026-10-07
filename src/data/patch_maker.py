import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def rle_decode(rle_str, shape=(256, 1600)):
    """Decodes RLE string into a 2D binary mask."""
    if pd.isna(rle_str) or not rle_str:
        return np.zeros(shape, dtype=np.uint8)
    s = rle_str.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order="F")

def mask_to_yolo_polygons(mask_patch, class_id, width, height, min_area=5):
    """
    Finds contours in a binary mask patch and converts them to YOLO polygon format.
    Coordinates are normalized between 0 and 1.
    """
    # Find contours
    contours, _ = cv2.findContours(mask_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_lines = []
    for contour in contours:
        # Filter out extremely tiny noise polygons
        if cv2.contourArea(contour) < min_area:
            continue
            
        # YOLO requires at least 3 points (6 coordinates) to form a polygon
        if contour.shape[0] >= 3:
            # Flatten the contour array
            contour = contour.flatten()
            
            # Normalize coordinates
            normalized_coords = []
            for i in range(len(contour)):
                if i % 2 == 0:  # X coordinate
                    normalized_coords.append(round(contour[i] / width, 5))
                else:           # Y coordinate
                    normalized_coords.append(round(contour[i] / height, 5))
            
            # Format: class_id x1 y1 x2 y2 ...
            coords_str = " ".join(map(str, normalized_coords))
            # Note: Serverstal classes are 1-4, YOLO needs 0-indexed classes (0-3)
            yolo_lines.append(f"{class_id - 1} {coords_str}")
            
    return yolo_lines

def process_dataset(csv_path, img_dir, out_images_dir, out_labels_dir, patch_width=400, patch_height=256):
    """
    Slices images and masks into patches and saves them in YOLOv8-seg format.
    """
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    image_groups = df.groupby("ImageId")
    
    # Standard Severstal image dimensions
    orig_w, orig_h = 1600, 256
    num_patches = orig_w // patch_width

    for image_id, group in tqdm(image_groups, desc="Processing Images"):
        img_path = os.path.join(img_dir, image_id)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Generate full multi-channel mask
        full_mask = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        for _, row in group.iterrows():
            c_id = int(row["ClassId"])
            if pd.notna(row["EncodedPixels"]):
                full_mask[:, :, c_id - 1] = rle_decode(row["EncodedPixels"], (orig_h, orig_w))

        # Slice into patches
        for i in range(num_patches):
            x_start = i * patch_width
            x_end = (i + 1) * patch_width
            
            # Crop image and mask
            img_patch = img[:, x_start:x_end]
            mask_patch = full_mask[:, x_start:x_end, :]
            
            patch_name = f"{os.path.splitext(image_id)[0]}_patch_{i}"
            
            # Extract polygons for YOLO
            patch_yolo_lines = []
            for c_idx in range(4):
                class_mask = mask_patch[:, :, c_idx]
                if np.max(class_mask) > 0:
                    lines = mask_to_yolo_polygons(class_mask, class_id=c_idx+1, width=patch_width, height=patch_height)
                    patch_yolo_lines.extend(lines)
            
            # Save the cropped image patch
            cv2.imwrite(os.path.join(out_images_dir, f"{patch_name}.jpg"), img_patch)
            
            # Save the YOLO labels (even if empty, to teach YOLO background images)
            with open(os.path.join(out_labels_dir, f"{patch_name}.txt"), "w") as f:
                f.write("\n".join(patch_yolo_lines))

if __name__ == "__main__":
    # Define your paths here
    CSV_PATH = "../../data/raw/train.csv"
    IMG_DIR = "../../data/raw/train_images"
    OUT_IMAGES = "../../data/processed/images"
    OUT_LABELS = "../../data/processed/labels"
    
    process_dataset(CSV_PATH, IMG_DIR, OUT_IMAGES, OUT_LABELS)