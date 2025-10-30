"""
Metrics Computation for Semantic Segmentation Validation

Implements IoU, mIoU, pixel accuracy, and F1 score calculations.
"""

import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMetrics:
    """
    Computes segmentation metrics for validation.
    
    Metrics computed:
    - Per-class IoU (Intersection over Union)
    - mIoU (mean IoU)
    - Pixel Accuracy
    - Per-class F1 Score
    - Mean F1 Score
    """
    
    def __init__(self, num_classes, ignore_index=255):
        """
        Args:
            num_classes (int): Number of segmentation classes
            ignore_index (int): Index to ignore in metric computation (e.g., background or void). Default: 255
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """
        Update metrics with new batch of predictions and targets.
        
        Args:
            predictions (torch.Tensor): Model predictions, shape (B, C, H, W) or (B, H, W)
            targets (torch.Tensor): Ground truth, shape (B, H, W)
        """
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)  # (B, H, W)
        
        # Move to CPU and convert to numpy
        predictions = predictions.cpu().numpy().astype(np.int32)
        targets = targets.cpu().numpy().astype(np.int32)
        
        # Flatten
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
        
        # Create mask to ignore certain indices
        mask = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[mask]
        targets = targets[mask]
        
        # Update confusion matrix
        for pred_class in range(self.num_classes):
            for target_class in range(self.num_classes):
                self.confusion_matrix[target_class, pred_class] += np.sum(
                    (predictions == pred_class) & (targets == target_class)
                )
    
    def compute_iou(self):
        """
        Compute per-class IoU and mIoU.
        
        Returns:
            tuple: (per_class_iou, mean_iou)
                per_class_iou: numpy array of shape (num_classes,)
                mean_iou: float, mean of per-class IoUs
        """
        # IoU = TP / (TP + FP + FN)
        # TP = diagonal of confusion matrix
        # FP = sum of column - TP
        # FN = sum of row - TP
        
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Avoid division by zero
        denominator = tp + fp + fn
        iou = np.where(denominator > 0, tp / denominator, 0.0)
        
        # Compute mean IoU (only for classes that appear in ground truth)
        valid_classes = denominator > 0
        if valid_classes.sum() > 0:
            mean_iou = iou[valid_classes].mean()
        else:
            mean_iou = 0.0
        
        return iou, mean_iou
    
    def compute_pixel_accuracy(self):
        """
        Compute overall pixel accuracy.
        
        Returns:
            float: Pixel accuracy (correct pixels / total pixels)
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        
        if total > 0:
            return correct / total
        else:
            return 0.0
    
    def compute_f1_score(self):
        """
        Compute per-class F1 score and mean F1.
        
        Returns:
            tuple: (per_class_f1, mean_f1)
                per_class_f1: numpy array of shape (num_classes,)
                mean_f1: float, mean of per-class F1 scores
        """
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        # Precision = TP / (TP + FP)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        
        # Recall = TP / (TP + FN)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        
        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = np.where(
            precision + recall > 0,
            2 * (precision * recall) / (precision + recall),
            0.0
        )
        
        # Compute mean F1 (only for classes that appear)
        valid_classes = (tp + fn) > 0
        if valid_classes.sum() > 0:
            mean_f1 = f1[valid_classes].mean()
        else:
            mean_f1 = 0.0
        
        return f1, mean_f1
    
    def compute_all(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        per_class_iou, mean_iou = self.compute_iou()
        pixel_acc = self.compute_pixel_accuracy()
        per_class_f1, mean_f1 = self.compute_f1_score()
        
        return {
            'per_class_iou': per_class_iou,
            'mean_iou': mean_iou,
            'pixel_accuracy': pixel_acc,
            'per_class_f1': per_class_f1,
            'mean_f1': mean_f1,
            'confusion_matrix': self.confusion_matrix
        }
    
    def get_summary_string(self, class_names=None):
        """
        Get a formatted string summary of metrics.
        
        Args:
            class_names (list): Optional list of class names for better readability
        
        Returns:
            str: Formatted summary string
        """
        metrics = self.compute_all()
        
        lines = []
        lines.append("=" * 80)
        lines.append("Validation Metrics Summary")
        lines.append("=" * 80)
        lines.append(f"Mean IoU:        {metrics['mean_iou']:.4f}")
        lines.append(f"Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
        lines.append(f"Mean F1 Score:   {metrics['mean_f1']:.4f}")
        lines.append("-" * 80)
        lines.append("Per-Class Metrics:")
        lines.append("-" * 80)
        
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"Class {i}"
            lines.append(
                f"  {class_name:20s}  IoU: {metrics['per_class_iou'][i]:.4f}  "
                f"F1: {metrics['per_class_f1'][i]:.4f}"
            )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


def validate_model(model, dataloader, device, num_classes, ignore_index=255):
    """
    Run validation on a model.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to run on
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in metrics
    
    Returns:
        dict: Dictionary containing validation metrics and average loss
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle batch format
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
                targets_dict = batch[1]
                
                # Extract segmentation mask
                if isinstance(targets_dict, dict):
                    if 'cls' in targets_dict:
                        targets = targets_dict['cls'].to(device)
                    else:
                        # Try other possible keys
                        targets = next(iter(targets_dict.values())).to(device)
                else:
                    targets = targets_dict.to(device)
            else:
                images = batch.to(device)
                targets = None
            
            # Forward pass
            outputs = model(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                # FarSeg/FarSegPP returns dict during training, but we need predictions
                # Run in eval mode to get predictions directly
                if 'semantic' in outputs:
                    predictions = outputs['semantic']
                elif 'seg' in outputs:
                    predictions = outputs['seg']
                else:
                    predictions = outputs[list(outputs.keys())[0]]
                
                # Compute loss if possible
                if 'loss' in outputs:
                    loss = outputs['loss']
                    if isinstance(loss, dict):
                        loss = sum(loss.values())
                    total_loss += loss.item()
                    num_batches += 1
            else:
                predictions = outputs
            
            # Update metrics if targets available
            if targets is not None:
                metrics.update(predictions, targets)
    
    # Compute final metrics
    result = metrics.compute_all()
    
    # Add average loss if computed
    if num_batches > 0:
        result['avg_loss'] = total_loss / num_batches
    
    return result
