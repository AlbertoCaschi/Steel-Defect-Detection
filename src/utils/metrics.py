import numpy as np

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice Similarity Coefficient (DSC) for a single class.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to prevent division by zero.
        
    Returns:
        float: Dice coefficient between 0 and 1.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Intersection over Union (IoU) metric.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to prevent division by zero.
        
    Returns:
        float: IoU score between 0 and 1.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    
    return (intersection + smooth) / (union + smooth)

def class_recall(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Recall (True Positive Rate), particularly useful 
    for tracking minority class performance.
    
    Args:
        y_true (np.ndarray): Ground truth binary mask.
        y_pred (np.ndarray): Predicted binary mask.
        smooth (float): Small value to prevent division by zero.
        
    Returns:
        float: Recall score between 0 and 1.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    true_positives = np.sum(y_true_f * y_pred_f)
    actual_positives = np.sum(y_true_f)
    
    # If there are no positive pixels in the ground truth, recall is undefined (return NaN)
    if actual_positives == 0:
        return np.nan
        
    return (true_positives + smooth) / (actual_positives + smooth)

def evaluate_batch(y_true_batch, y_pred_batch, num_classes=4):
    """
    Evaluates a batch of multi-channel masks across all metrics.
    
    Args:
        y_true_batch (np.ndarray): Ground truth masks, shape (Batch, Classes, H, W).
        y_pred_batch (np.ndarray): Predicted masks, shape (Batch, Classes, H, W).
        num_classes (int): Number of defect classes.
        
    Returns:
        dict: Aggregated Dice, IoU, and Recall per class.
    """
    results = {}
    
    for class_idx in range(num_classes):
        class_true = y_true_batch[:, class_idx, :, :]
        class_pred = y_pred_batch[:, class_idx, :, :]
        
        dice = dice_coefficient(class_true, class_pred)
        iou = iou_score(class_true, class_pred)
        recall = class_recall(class_true, class_pred)
        
        results[f"Class_{class_idx+1}"] = {
            "Dice": round(dice, 4),
            "IoU": round(iou, 4),
            "Recall": round(recall, 4) if not np.isnan(recall) else None
        }
        
    return results

if __name__ == "__main__":
    # Simulate a test case
    np.random.seed(42)
    # Batch size 1, 4 classes, 256x256 resolution
    dummy_true = np.random.randint(0, 2, size=(1, 4, 256, 256))
    
    # Perfect prediction for class 1, random for others
    dummy_pred = dummy_true.copy()
    dummy_pred[:, 1:, :, :] = np.random.randint(0, 2, size=(1, 3, 256, 256))
    
    metrics = evaluate_batch(dummy_true, dummy_pred)
    
    print("Evaluation Results:")
    for cls, scores in metrics.items():
        print(f"{cls}: {scores}")