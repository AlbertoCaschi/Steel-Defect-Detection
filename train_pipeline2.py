import os
import argparse
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.models.yolo_wrapper import YOLOWrapper


def generate_yolo_dataset_yaml(
    data_dir,
    output_yaml_path="configs/yolo_seg_data.yaml",
    val_ratio=0.2,
    seed=42
):
    """
    Splits tiled image patches into train/val subsets, writes split manifest files,
    and produces the dataset YAML file required by Ultralytics YOLO segmentation.
    """
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    # Gather all image files
    all_images = sorted([
        str(p.resolve()) for p in images_dir.glob("*.jpg")
        if (labels_dir / f"{p.stem}.txt").exists()
    ])

    if len(all_images) == 0:
        raise RuntimeError(f"No valid image-label pairs found in {data_dir}")

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=val_ratio, random_state=seed, shuffle=True
    )

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    train_txt = split_dir / "train_seg.txt"
    val_txt = split_dir / "val_seg.txt"

    with open(train_txt, "w") as f:
        f.write("\n".join(train_imgs) + "\n")

    with open(val_txt, "w") as f:
        f.write("\n".join(val_imgs) + "\n")

    dataset_dict = {
        "path": str(Path.cwd()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": 4,
        "names": {
            0: "pitted_surface",
            1: "crazing",
            2: "scratch",
            3: "patch"
        }
    }

    Path(output_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(dataset_dict, f, sort_keys=False)

    print(f"Generated dataset config at: {output_yaml_path}")
    print(f"Total tiles: {len(all_images)} (Train: {len(train_imgs)}, Val: {len(val_imgs)})")

    return output_yaml_path


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "pipeline2_baseline")
    model_weights = cfg.get("yolo_model", "yolov8n-seg.pt")
    data_dir = cfg.get("processed_data_dir", "data/processed")
    epochs = cfg.get("epochs", 50)
    batch_size = cfg.get("batch_size", 16)
    imgsz = cfg.get("imgsz", 400)
    device = cfg.get("device", 0)
    workers = cfg.get("workers", 4)
    optimizer = cfg.get("optimizer", "AdamW")
    lr0 = cfg.get("lr0", 1e-3)
    patience = cfg.get("patience", 15)

    print(f"=== Starting Pipeline 2 (One-Stage YOLOv8-seg): {exp_name} ===")

    # Prepare or retrieve YOLO data YAML
    data_yaml = cfg.get("yolo_data_yaml")
    if not data_yaml or not os.path.exists(data_yaml):
        data_yaml = generate_yolo_dataset_yaml(
            data_dir=data_dir,
            output_yaml_path=f"configs/data_{exp_name}.yaml",
            val_ratio=cfg.get("val_ratio", 0.2),
            seed=cfg.get("seed", 42)
        )

    # Initialize YOLO segmentation wrapper
    yolo = YOLOWrapper(model_name_or_path=model_weights)

    # Train model
    train_results = yolo.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        project="runs/pipeline2",
        name=exp_name,
        save=True,
        plots=True
    )

    # Evaluate validation split (extracting box and mask metrics)
    print("\n--- Running Final Validation ---")
    val_metrics = yolo.evaluate(
        data_yaml=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=0.25
    )

    # Print mask and box summary
    if hasattr(val_metrics, "seg"):
        print("\nMask Segmentation Metrics:")
        print(f"  Mask mAP50:     {val_metrics.seg.map50:.4f}")
        print(f"  Mask mAP50-95:  {val_metrics.seg.map:.4f}")
    if hasattr(val_metrics, "box"):
        print("\nBounding Box Metrics:")
        print(f"  Box mAP50:      {val_metrics.box.map50:.4f}")
        print(f"  Box mAP50-95:   {val_metrics.box.map:.4f}")

    print(f"\nTraining complete. Artifacts stored in runs/pipeline2/{exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pipeline 2: YOLOv8-seg on Tiled Steel Patches")
    parser.add_argument("--config", type=str, default="configs/pipeline2_baseline.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # Generate a sample default configuration file if absent
    if not os.path.exists(args.config):
        os.makedirs("configs", exist_ok=True)
        default_cfg = {
            "experiment_name": "yolov8_seg_tiled_baseline",
            "yolo_model": "yolov8n-seg.pt",
            "processed_data_dir": "data/processed",
            "yolo_data_yaml": "configs/yolo_seg_data.yaml",
            "epochs": 50,
            "batch_size": 16,
            "imgsz": 400,
            "device": 0,
            "workers": 4,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "patience": 15,
            "val_ratio": 0.2,
            "seed": 42
        }
        with open(args.config, "w") as f:
            yaml.safe_dump(default_cfg, f, sort_keys=False)
        print(f"Created default configuration: {args.config}")

    main(args.config)