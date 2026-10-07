import os
from ultralytics import YOLO

class YOLOWrapper:
    """
    A unified wrapper for Ultralytics YOLO models supporting both 
    bounding box detection (Pipeline 1) and instance segmentation (Pipeline 2).
    """
    def __init__(self, model_name_or_path="yolo11n.pt"):
        """
        Initializes the YOLO model.
        
        Args:
            model_name_or_path (str): The model to load. 
                - Use 'yolo11n.pt' for Pipeline 1 (Detection).
                - Use 'yolo11n-seg.pt' for Pipeline 2 (Segmentation).
                - Can also be a path to a custom trained .pt file.
        """
        print(f"Loading YOLO model from: {model_name_or_path}")
        self.model = YOLO(model_name_or_path)

    def train(self, data_yaml, epochs=50, imgsz=400, batch=16, project="runs", name="yolo_exp", **kwargs):
        """
        Trains the YOLO model.
        
        Args:
            data_yaml (str): Path to the dataset YAML configuration file.
            epochs (int): Number of training epochs.
            imgsz (int): Target image size for training.
            batch (int): Batch size.
            project (str): Directory to save training runs.
            name (str): Name of the current training experiment.
            **kwargs: Additional YOLO training arguments (e.g., lr0, patience, optimizer).
            
        Returns:
            The training results object.
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

        print(f"Starting training on {data_yaml} for {epochs} epochs...")
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project,
            name=name,
            **kwargs
        )
        return results

    def evaluate(self, data_yaml, split="val", imgsz=400, conf=0.25):
        """
        Evaluates the model on a validation or test set.
        
        Args:
            data_yaml (str): Path to the dataset YAML config.
            split (str): Split to evaluate on ('val' or 'test').
            imgsz (int): Image size for evaluation.
            conf (float): Confidence threshold for predictions.
            
        Returns:
            The validation metrics object (contains mAP50, mAP50-95, etc.).
        """
        print(f"Evaluating model on '{split}' split...")
        metrics = self.model.val(
            data=data_yaml,
            split=split,
            imgsz=imgsz,
            conf=conf
        )
        return metrics

    def predict(self, source, conf=0.25, save=False, project="runs", name="predict", **kwargs):
        """
        Runs inference on the provided source.
        
        Args:
            source (str, list, np.ndarray): Image path, directory, or array to run inference on.
            conf (float): Confidence threshold.
            save (bool): Whether to save the annotated images to disk.
            project (str): Directory to save prediction results.
            name (str): Folder name for the current predictions.
            **kwargs: Additional inference arguments.
            
        Returns:
            A list of Ultralytics Results objects.
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            save=save,
            project=project,
            name=name,
            **kwargs
        )
        return results

    def export(self, format="onnx", imgsz=400):
        """
        Exports the trained model to a different format for deployment.
        
        Args:
            format (str): Target format (e.g., 'onnx', 'engine', 'torchscript').
            imgsz (int): Image size expected by the exported model.
            
        Returns:
            The path to the exported model.
        """
        print(f"Exporting model to {format} format...")
        export_path = self.model.export(format=format, imgsz=imgsz)
        return export_path


if __name__ == "__main__":
    # Example usage for testing the wrapper instantiation
    print("--- Testing Detection Wrapper (Pipeline 1) ---")
    det_wrapper = YOLOWrapper("yolo11n.pt")
    
    print("\n--- Testing Segmentation Wrapper (Pipeline 2) ---")
    seg_wrapper = YOLOWrapper("yolo11n-seg.pt")