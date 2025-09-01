"""
BuildFormer-style evaluator for segmentation metrics.
This implements the global confusion matrix accumulation approach.
"""
import numpy as np


class Evaluator(object):
    """
    Segmentation evaluator using global confusion matrix accumulation.
    This follows the same approach as BuildFormer's tools/metric.py.
    """
    
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        """Extract TP, FP, TN, FN from confusion matrix."""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        """Calculate precision per class - exact BuildFormer implementation."""
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        """Calculate recall per class - exact BuildFormer implementation."""
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        """Calculate F1 score per class - exact BuildFormer implementation."""
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        """Calculate Overall Accuracy."""
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        """
        Calculate IoU per class using global confusion matrix.
        This is the standard approach: IoU = TP / (TP + FN + FP)
        Exact BuildFormer implementation.
        """
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        """Calculate Dice coefficient per class - exact BuildFormer implementation."""
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        """Calculate pixel accuracy per class."""
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        """Calculate frequency-weighted IoU."""
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """Generate confusion matrix for a single sample."""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """
        Add a batch of predictions to the global confusion matrix.
        
        Args:
            gt_image: Ground truth mask
            pre_image: Predicted mask
        """
        assert gt_image.shape == pre_image.shape, f'pre_image shape {pre_image.shape}, gt_image shape {gt_image.shape}'
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        """Reset the confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def summary(self):
        """
        Get summary of all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics (same as BuildFormer)
        """
        iou_per_class = self.Intersection_over_Union()
        f1_per_class = self.F1()
        precision_per_class = self.Precision()
        recall_per_class = self.Recall()
        oa = self.OA()
        dice_per_class = self.Dice()
        pixel_accuracy_per_class = self.Pixel_Accuracy_Class()
        fwiou = self.Frequency_Weighted_Intersection_over_Union()
        
        # Mean IoU (excluding background class if desired)
        miou = np.nanmean(iou_per_class)
        
        return {
            'iou_per_class': iou_per_class,
            'miou': miou,
            'f1_per_class': f1_per_class,
            'mean_f1': np.nanmean(f1_per_class),
            'precision_per_class': precision_per_class,
            'mean_precision': np.nanmean(precision_per_class),
            'recall_per_class': recall_per_class,
            'mean_recall': np.nanmean(recall_per_class),
            'dice_per_class': dice_per_class,
            'mean_dice': np.nanmean(dice_per_class),
            'pixel_accuracy_per_class': pixel_accuracy_per_class,
            'mean_pixel_accuracy': np.nanmean(pixel_accuracy_per_class),
            'overall_accuracy': oa,
            'frequency_weighted_iou': fwiou,
            'confusion_matrix': self.confusion_matrix
        }


if __name__ == '__main__':
    # Example usage (same as BuildFormer)
    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(gt, pre)
    
    print("Confusion Matrix:")
    print(evaluator.confusion_matrix)
    print("\nTP, FP, TN, FN:")
    print(evaluator.get_tp_fp_tn_fn())
    print(f"\nPrecision: {evaluator.Precision()}")
    print(f"Recall: {evaluator.Recall()}")
    print(f"IoU: {evaluator.Intersection_over_Union()}")
    print(f"Overall Accuracy: {evaluator.OA()}")
    print(f"F1: {evaluator.F1()}")
    print(f"Dice: {evaluator.Dice()}")
    print(f"Pixel Accuracy Class: {evaluator.Pixel_Accuracy_Class()}")
    print(f"Frequency Weighted IoU: {evaluator.Frequency_Weighted_Intersection_over_Union()}")
    
    print("\n" + "="*50)
    print("SUMMARY OF ALL METRICS:")
    summary = evaluator.summary()
    for key, value in summary.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")
