import os
import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import custom modules from your project structure
from src.models.yolo_wrapper import YOLOWrapper
from src.models.unet import build_unet
from src.data.dataset import get_transforms, rle_decode
from src.utils.geometry import apply_margin_to_bbox, crop_for_unet

# ---------------------------------------------------------------------------
# 1. Stage 2 Data Preparation: Cropping GT for U-Net
# ---------------------------------------------------------------------------
def prepare_unet_crops(csv_path, img_dir, out_crop_dir, margin=10, target_size=(256, 256)):
    """
    Extracts ground-truth bounding boxes from the RLE masks, applies the margin,
    and saves the cropped images and cropped masks for U-Net training.
    """
    print(f"Preparing U-Net dataset crops in {out_crop_dir}...")
    crop_img_dir = os.path.join(out_crop_dir, "images")
    crop_mask_dir = os.path.join(out_crop_dir, "masks")
    os.makedirs(crop_img_dir, exist_ok=True)
    os.makedirs(crop_mask_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    defect_df = df.dropna(subset=["EncodedPixels"])
    
    crop_records = []
    
    for idx, row in tqdm(defect_df.iterrows(), total=len(defect_df), desc="Cropping"):
        img_name = row["ImageId"]
        class_id = int(row["ClassId"])
        img_path = os.path.join(img_dir, img_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Decode mask and find bounding box
        mask = rle_decode(row["EncodedPixels"], shape=(256, 1600))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        # For simplicity, grab the bounding box of the largest defect component
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Apply Margin using geometry.py
        padded_bbox = apply_margin_to_bbox((x, y, x+w, y+h), margin=margin)
        
        # Crop Image and Mask
        cropped_img = crop_for_unet(img, padded_bbox, target_size=target_size)
        cropped_mask = crop_for_unet(mask, padded_bbox, target_size=target_size)
        
        # Save crops
        crop_id = f"{os.path.splitext(img_name)[0]}_cls{class_id}_{idx}"
        img_save_path = os.path.join(crop_img_dir, f"{crop_id}.jpg")
        mask_save_path = os.path.join(crop_mask_dir, f"{crop_id}.npy")
        
        cv2.imwrite(img_save_path, cropped_img)
        np.save(mask_save_path, cropped_mask)
        
        crop_records.append({
            "crop_id": crop_id,
            "class_id": class_id,
            "img_path": img_save_path,
            "mask_path": mask_save_path
        })
        
    return pd.DataFrame(crop_records)

# ---------------------------------------------------------------------------
# 2. U-Net Dataset Class (For Cropped Patches)
# ---------------------------------------------------------------------------
class UNetCropDataset(Dataset):
    def __init__(self, crop_df, transform=None):
        self.crop_df = crop_df
        self.transform = transform

    def __len__(self):
        return len(self.crop_df)

    def __getitem__(self, idx):
        row = self.crop_df.iloc[idx]
        image = cv2.imread(row["img_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load binary mask and reshape to (H, W, 1) for Albumentations
        mask = np.load(row["mask_path"])
        mask = np.expand_dims(mask, axis=-1).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            # Ensure mask is (1, H, W)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
            else:
                mask = mask.permute(2, 0, 1)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return image, mask, row["class_id"]

# ---------------------------------------------------------------------------
# 3. Main Training Orchestrator
# ---------------------------------------------------------------------------
def main(config_path="configs/exp0_baseline.yaml"):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"--- Starting Pipeline 1 Training: {config['experiment_name']} ---")

    # ==========================================
    # STAGE 1: Train YOLOv11 (Bounding Box)
    # ==========================================
    if config.get("train_yolo", True):
        print("\n[Stage 1] Training YOLOv11 Detector...")
        yolo = YOLOWrapper("yolo11n.pt")
        yolo.train(
            data_yaml=config["yolo_data_yaml"],
            epochs=config["yolo_epochs"],
            imgsz=config["yolo_imgsz"],
            batch=config["yolo_batch"],
            project="runs/pipeline1",
            name=f"yolo_{config['experiment_name']}"
        )
    else:
        print("\n[Stage 1] Skipping YOLOv11 Training (using pre-trained).")

    # ==========================================
    # STAGE 2: Train U-Net (Segmentation)
    # ==========================================
    if config.get("train_unet", True):
        print("\n[Stage 2] Preparing Data & Training U-Net...")
        
        # 2a. Prepare Crops
        crop_dir = os.path.join(config["processed_data_dir"], "unet_crops")
        if not os.path.exists(crop_dir):
            crop_df = prepare_unet_crops(
                csv_path=config["train_csv"],
                img_dir=config["train_img_dir"],
                out_crop_dir=crop_dir,
                margin=config["crop_margin"],
                target_size=(256, 256)
            )
            crop_df.to_csv(os.path.join(crop_dir, "crop_manifest.csv"), index=False)
        else:
            print("Crops already exist. Loading manifest...")
            crop_df = pd.read_csv(os.path.join(crop_dir, "crop_manifest.csv"))

        # 2b. Initialize Dataset & DataLoader
        dataset = UNetCropDataset(crop_df, transform=get_transforms("train"))
        dataloader = DataLoader(dataset, batch_size=config["unet_batch"], shuffle=True, num_workers=4)

        # 2c. Initialize Model, Loss, Optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_unet(
            encoder_name="efficientnet-b0", 
            in_channels=3, 
            num_classes=1, # 1 class per crop (binary defect inside the ROI)
            use_smp=True
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["unet_lr"])

        # 2d. U-Net Training Loop
        epochs = config["unet_epochs"]
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"U-Net Epoch {epoch+1}/{epochs}")
            for images, masks, _ in pbar:
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
            
        # Save U-Net Weights
        os.makedirs("runs/pipeline1/unet_weights", exist_ok=True)
        torch.save(model.state_dict(), f"runs/pipeline1/unet_weights/{config['experiment_name']}_unet.pth")
        print("\n[Pipeline 1] Training Complete.")

if __name__ == "__main__":
    # Mocking a basic config directly for standalone testing without a YAML file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp0_baseline.yaml")
    args = parser.parse_args()
    
    # Normally you would load args.config, but here is a fallback dict for execution
    if not os.path.exists(args.config):
        print(f"Config {args.config} not found. Running with default dummy config.")
        dummy_config = {
            "experiment_name": "baseline",
            "train_yolo": False, # Set to True when you have YOLO data ready
            "yolo_data_yaml": "data/yolo_config.yaml",
            "yolo_epochs": 10,
            "yolo_imgsz": 400,
            "yolo_batch": 16,
            "train_unet": True,
            "train_csv": "dataset/train.csv",
            "train_img_dir": "dataset/train_images",
            "processed_data_dir": "data/processed",
            "crop_margin": 10,
            "unet_batch": 16,
            "unet_lr": 1e-4,
            "unet_epochs": 10
        }
        # Dump dummy to file to allow script to run
        os.makedirs("configs", exist_ok=True)
        with open(args.config, "w") as f:
            yaml.dump(dummy_config, f)
            
    main(args.config)