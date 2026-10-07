import os
import cv2
import time
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.yolo_wrapper import YOLOWrapper
from src.models.unet import build_unet
from src.data.dataset import rle_decode
from src.utils.geometry import apply_margin_to_bbox, crop_for_unet, restore_full_mask
from src.utils.metrics import dice_coefficient, class_recall

# ---------------------------------------------------------------------------
# Pipeline 1 Inference (YOLOv11 Detect -> Crop -> U-Net)
# ---------------------------------------------------------------------------
def run_pipeline1(image, yolo_det, unet, device, margin=10, unet_size=(256, 256)):
    """Runs the two-stage cascade and returns a full-resolution 4-channel mask."""
    full_mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    
    # 1. YOLO Detection
    results = yolo_det.predict(image, verbose=False, conf=0.25)[0]
    
    if len(results.boxes) == 0:
        return full_mask
        
    for box in results.boxes:
        # Extract box and class
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].item()) 
        
        # 2. Crop and apply margin
        padded_box = apply_margin_to_bbox((x1, y1, x2, y2), margin=margin)
        crop_img = crop_for_unet(image, padded_box, target_size=unet_size)
        
        # Prepare for U-Net (Normalize)
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_tensor = torch.tensor(crop_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Basic ImageNet normalization used during training
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        crop_tensor = (crop_tensor - mean) / std
        crop_tensor = crop_tensor.unsqueeze(0).to(device)
        
        # 3. U-Net Segmentation
        with torch.no_grad():
            logits = unet(crop_tensor)
            pred_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            binary_pred = (pred_prob > 0.5).astype(np.uint8)
            
        # 4. Map back to full resolution
        restored_mask = restore_full_mask(binary_pred, padded_box, original_shape=(256, 1600))
        
        # Combine if multiple boxes exist for the same class
        full_mask[:, :, cls_id] = np.logical_or(full_mask[:, :, cls_id], restored_mask).astype(np.uint8)
        
    return full_mask

# ---------------------------------------------------------------------------
# Pipeline 2 Inference (Tiled YOLOv8-seg)
# ---------------------------------------------------------------------------
def run_pipeline2(image, yolo_seg, patch_w=400, patch_h=256):
    """Slices the image, runs YOLO segmentation on patches, and stitches them back."""
    full_mask = np.zeros((256, 1600, 4), dtype=np.uint8)
    num_patches = 1600 // patch_w
    
    patches = []
    for i in range(num_patches):
        x_start = i * patch_w
        x_end = (i + 1) * patch_w
        patches.append(image[:, x_start:x_end])
        
    # Batch predict patches for faster inference
    results = yolo_seg.predict(patches, verbose=False, conf=0.25)
    
    for i, res in enumerate(results):
        if res.masks is not None:
            # Ultralytics masks are resized to original image shape (here 256x400)
            masks_data = res.masks.data.cpu().numpy()
            # Original shapes might need resizing if YOLO scaled them internally
            if masks_data.shape[1:] != (patch_h, patch_w):
                masks_data = np.array([cv2.resize(m, (patch_w, patch_h)) for m in masks_data])
                
            masks_data = (masks_data > 0.5).astype(np.uint8)
            
            x_start = i * patch_w
            x_end = (i + 1) * patch_w
            
            for j, box in enumerate(res.boxes):
                cls_id = int(box.cls[0].item())
                patch_mask = masks_data[j]
                
                # Overlay onto the full image canvas
                full_mask[:, x_start:x_end, cls_id] = np.logical_or(
                    full_mask[:, x_start:x_end, cls_id], patch_mask
                ).astype(np.uint8)
                
    return full_mask

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
def main(config_path="configs/evaluate.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    print("=== Severstal Steel Defect Pipeline Benchmark ===")
    
    # 1. Initialize Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading Pipeline 1 (Two-Stage)...")
    yolo_det = YOLOWrapper(cfg["p1_yolo_weights"])
    unet = build_unet(encoder_name="efficientnet-b0", in_channels=3, num_classes=1, use_smp=True).to(device)
    unet.load_state_dict(torch.load(cfg["p1_unet_weights"], map_location=device))
    unet.eval()
    
    print("Loading Pipeline 2 (One-Stage)...")
    yolo_seg = YOLOWrapper(cfg["p2_yolo_weights"])
    
    # 2. Prepare Dataset
    df = pd.read_csv(cfg["test_csv"])
    test_images = df["ImageId"].unique()[:cfg.get("num_eval_samples", 500)] # Limit for quick testing
    img_dir = cfg["test_img_dir"]
    
    # Tracking Dictionaries
    metrics = {
        "P1": {"dice": [], "recall_c1": [], "recall_c2": [], "recall_c3": [], "recall_c4": [], "time": []},
        "P2": {"dice": [], "recall_c1": [], "recall_c2": [], "recall_c3": [], "recall_c4": [], "time": []}
    }
    
    print(f"\nRunning benchmark on {len(test_images)} images...")
    
    for img_name in tqdm(test_images):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Build Ground Truth Full Mask
        gt_mask = np.zeros((256, 1600, 4), dtype=np.uint8)
        img_df = df[df["ImageId"] == img_name]
        for _, row in img_df.iterrows():
            if pd.notna(row["EncodedPixels"]):
                cls_id = int(row["ClassId"]) - 1
                gt_mask[:, :, cls_id] = rle_decode(row["EncodedPixels"], shape=(256, 1600))
                
        # --- Evaluate Pipeline 1 ---
        start_t = time.perf_counter()
        p1_mask = run_pipeline1(image, yolo_det, unet, device, margin=cfg["crop_margin"])
        metrics["P1"]["time"].append(time.perf_counter() - start_t)
        
        # --- Evaluate Pipeline 2 ---
        start_t = time.perf_counter()
        p2_mask = run_pipeline2(image, yolo_seg)
        metrics["P2"]["time"].append(time.perf_counter() - start_t)
        
        # --- Calculate Metrics ---
        for pipe_name, pred_mask in zip(["P1", "P2"], [p1_mask, p2_mask]):
            for c in range(4):
                gt_c = gt_mask[:, :, c]
                pred_c = pred_mask[:, :, c]
                
                # Only calculate Dice if there's GT or Prediction (ignore true negatives)
                if np.sum(gt_c) > 0 or np.sum(pred_c) > 0:
                    metrics[pipe_name]["dice"].append(dice_coefficient(gt_c, pred_c))
                
                # Only calculate Recall if GT exists
                if np.sum(gt_c) > 0:
                    metrics[pipe_name][f"recall_c{c+1}"].append(class_recall(gt_c, pred_c))

    # 3. Aggregate and Print Results
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Pipeline 1 (2-Stage)':<20} | {'Pipeline 2 (1-Stage)':<20}")
    print("="*50)
    
    def agg(m_list): return np.nanmean(m_list) if len(m_list) > 0 else 0.0
    
    p1_fps = 1.0 / agg(metrics['P1']['time'])
    p2_fps = 1.0 / agg(metrics['P2']['time'])
    
    print(f"{'Mean Dice Score':<20} | {agg(metrics['P1']['dice']):<20.4f} | {agg(metrics['P2']['dice']):<20.4f}")
    print("-" * 50)
    for c in range(1, 5):
        key = f"recall_c{c}"
        print(f"{f'Recall Class {c}':<20} | {agg(metrics['P1'][key]):<20.4f} | {agg(metrics['P2'][key]):<20.4f}")
    print("-" * 50)
    print(f"{'Inference Speed':<20} | {p1_fps:<14.2f} FPS   | {p2_fps:<14.2f} FPS")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate.yaml")
    args = parser.parse_args()
    
    # Generate dummy config if missing
    if not os.path.exists(args.config):
        os.makedirs("configs", exist_ok=True)
        dummy_cfg = {
            "p1_yolo_weights": "runs/pipeline1/yolo_baseline/weights/best.pt",
            "p1_unet_weights": "runs/pipeline1/unet_weights/baseline_unet.pth",
            "p2_yolo_weights": "runs/pipeline2/yolov8_seg_tiled_baseline/weights/best.pt",
            "test_csv": "dataset/train.csv",      # Using train.csv as test manifest for demo
            "test_img_dir": "dataset/train_images",
            "crop_margin": 10,
            "num_eval_samples": 200 # Run on a subset to save time
        }
        with open(args.config, "w") as f:
            yaml.safe_dump(dummy_cfg, f)
            
    main(args.config)