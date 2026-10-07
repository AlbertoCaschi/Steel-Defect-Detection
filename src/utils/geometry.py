import numpy as np
import cv2

def apply_margin_to_bbox(bbox, img_width=1600, img_height=256, margin=10):
    """
    Applies a pixel margin around a bounding box while ensuring the new 
    coordinates do not exceed the image boundaries.
    
    Args:
        bbox (tuple): Original bounding box (x_min, y_min, x_max, y_max).
        img_width (int): Maximum image width.
        img_height (int): Maximum image height.
        margin (int): Number of pixels to expand the box by.
        
    Returns:
        tuple: New bounding box (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Apply margin and clip to image boundaries
    new_x_min = max(0, int(x_min - margin))
    new_y_min = max(0, int(y_min - margin))
    new_x_max = min(img_width, int(x_max + margin))
    new_y_max = min(img_height, int(y_max + margin))
    
    return (new_x_min, new_y_min, new_x_max, new_y_max)


def crop_for_unet(image, bbox, target_size=(256, 256)):
    """
    Crops the image using the provided bounding box and resizes it 
    for the U-Net input.
    
    Args:
        image (np.ndarray): Original full-resolution image.
        bbox (tuple): Bounding box (x_min, y_min, x_max, y_max) with margins applied.
        target_size (tuple): The expected input size for U-Net (W, H).
        
    Returns:
        np.ndarray: The cropped and resized image patch.
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Crop the region of interest
    crop = image[y_min:y_max, x_min:x_max]
    
    # Resize to U-Net input dimensions (256x256)
    if crop.size > 0:
        crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        crop_resized = np.zeros((target_size[1], target_size[0], image.shape[2]), dtype=image.dtype)
        
    return crop_resized


def restore_full_mask(unet_mask, bbox, original_shape=(256, 1600)):
    """
    Maps a predicted U-Net mask (e.g., 256x256) back to its exact location 
    on the full-resolution steel image.
    
    Args:
        unet_mask (np.ndarray): The output mask from U-Net (H, W).
        bbox (tuple): The margin-applied bounding box (x_min, y_min, x_max, y_max).
        original_shape (tuple): The shape of the original image (H, W).
        
    Returns:
        np.ndarray: A full-resolution binary mask.
    """
    x_min, y_min, x_max, y_max = bbox
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    
    # Create an empty canvas for the full resolution mask
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    
    # Resize the U-Net prediction back to the true bounding box dimensions
    if crop_w > 0 and crop_h > 0:
        # cv2.resize expects (width, height)
        mask_restored = cv2.resize(unet_mask.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        
        # Binarize in case interpolation caused floating point smoothing
        mask_restored = (mask_restored > 0.5).astype(np.uint8)
        
        # Paste the restored mask into the correct location on the full canvas
        full_mask[y_min:y_max, x_min:x_max] = mask_restored
        
    return full_mask


if __name__ == "__main__":
    # Test the geometric mapping logic
    original_img_shape = (256, 1600)
    
    # Example YOLO detection (x_min, y_min, x_max, y_max)
    yolo_box = (150, 50, 300, 100)
    
    # 1. Apply 10-pixel margin
    padded_box = apply_margin_to_bbox(yolo_box, margin=10)
    print(f"Original Box: {yolo_box}")
    print(f"Padded Box:   {padded_box}")
    
    # 2. Simulate U-Net prediction (256x256 crop)
    dummy_unet_prediction = np.zeros((256, 256), dtype=np.uint8)
    # Simulate a detected defect in the middle of the crop
    dummy_unet_prediction[100:150, 100:150] = 1
    
    # 3. Restore to 1600x256
    final_mask = restore_full_mask(dummy_unet_prediction, padded_box, original_shape=original_img_shape)
    print(f"Restored Mask Shape: {final_mask.shape}")
    print(f"Non-zero pixels mapped successfully: {np.sum(final_mask) > 0}")