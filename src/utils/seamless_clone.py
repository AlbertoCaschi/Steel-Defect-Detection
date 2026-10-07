import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

def rle_decode(rle_str, shape=(256, 1600)):
    """Decodes RLE string into a 2D binary mask."""
    s = rle_str.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order="F")

def rle_encode(mask):
    """Encodes a 2D binary mask back into an RLE string."""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def extract_defect_component(image, mask):
    """
    Extracts the bounding box crop of the largest connected defect 
    in the mask to prepare for seamless cloning.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
        
    # Select the largest defect region to ensure high-quality cloning
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Filter out defects that are too small or span the entire image
    if w < 10 or h < 10 or w > 800:
        return None, None, None
        
    defect_crop = image[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    
    return defect_crop, mask_crop, (w, h)

def generate_augmented_dataset(csv_path, img_dir, out_img_dir, out_csv_path, num_samples=1000, target_classes=[1, 2]):
    """
    Generates synthetic training images by extracting target minority classes 
    and seamlessly cloning them onto defect-free backgrounds.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Separate defect-free images (backgrounds) from defective images
    all_images = set(os.listdir(img_dir))
    defective_images = set(df["ImageId"].unique())
    clean_images = list(all_images - defective_images)
    
    # Filter source defects to only our target minority classes
    minority_df = df[(df["ClassId"].isin(target_classes)) & (df["EncodedPixels"].notna())]
    
    augmented_rows = []
    
    print(f"Generating {num_samples} augmented samples...")
    for i in tqdm(range(num_samples)):
        # 1. Randomly select a source defect
        source_row = minority_df.sample(1).iloc[0]
        source_img_path = os.path.join(img_dir, source_row["ImageId"])
        source_img = cv2.imread(source_img_path)
        source_mask = rle_decode(source_row["EncodedPixels"])
        
        defect_img, defect_mask, dims = extract_defect_component(source_img, source_mask)
        if defect_img is None:
            continue
            
        w, h = dims
        
        # 2. Randomly select a clean background
        bg_name = random.choice(clean_images)
        bg_img = cv2.imread(os.path.join(img_dir, bg_name))
        
        # 3. Determine a valid random center point for pasting
        # Ensure the defect doesn't cross the image boundaries
        center_x = random.randint(w // 2 + 10, bg_img.shape[1] - w // 2 - 10)
        center_y = random.randint(h // 2 + 10, bg_img.shape[0] - h // 2 - 10)
        center = (center_x, center_y)
        
        # 4. Perform Seamless Cloning
        # cv2.MIXED_CLONE often works best for steel textures, preserving background grain
        clone_mask = (defect_mask * 255).astype(np.uint8)
        try:
            blended_img = cv2.seamlessClone(defect_img, bg_img, clone_mask, center, cv2.MIXED_CLONE)
        except Exception as e:
            # OpenCV solver occasionally fails if gradients are too extreme
            continue
            
        # 5. Generate the new mask for the augmented image
        new_full_mask = np.zeros((256, 1600), dtype=np.uint8)
        # Calculate bounding box in the new image
        start_x = center_x - w // 2
        start_y = center_y - h // 2
        
        # Overlay the mask crop onto the new full mask
        new_full_mask[start_y:start_y+h, start_x:start_x+w] = defect_mask
        
        # 6. Save the new image and annotation
        new_img_name = f"aug_mix_{i}_{bg_name}"
        cv2.imwrite(os.path.join(out_img_dir, new_img_name), blended_img)
        
        augmented_rows.append({
            "ImageId": new_img_name,
            "ClassId": source_row["ClassId"],
            "EncodedPixels": rle_encode(new_full_mask)
        })
        
    # Combine original annotations with the new augmented annotations
    aug_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    combined_df.to_csv(out_csv_path, index=False)
    
    print(f"Saved {len(augmented_rows)} augmented images to {out_img_dir}")
    print(f"Updated annotations saved to {out_csv_path}")

if __name__ == "__main__":
    # Adjust paths based on your project structure
    CSV_PATH = "../../dataset/train.csv"
    IMG_DIR = "../../dataset/train_images"
    OUT_IMG_DIR = "../../data/augmented/images"
    OUT_CSV_PATH = "../../data/augmented/train_augmented.csv"
    
    # Generate 1500 new samples targeting Class 1 (Pitted) and Class 2 (Crazing)
    generate_augmented_dataset(
        csv_path=CSV_PATH, 
        img_dir=IMG_DIR, 
        out_img_dir=OUT_IMG_DIR, 
        out_csv_path=OUT_CSV_PATH,
        num_samples=1500,
        target_classes=[1, 2]
    )