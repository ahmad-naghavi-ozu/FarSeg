"""
Early Stopping Implementation for FarSeg/FarSeg++ Training

Monitors validation metrics and stops training when improvement plateaus.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    
    Args:
        patience (int): How many epochs to wait after last improvement. Default: 10
        min_delta (float): Minimum change in monitored metric to qualify as improvement. Default: 0.001
        mode (str): One of 'min' or 'max'. In 'min' mode, training stops when metric stops decreasing;
                   in 'max' mode it stops when metric stops increasing. Default: 'max'
        verbose (bool): If True, prints messages. Default: True
        restore_best_weights (bool): Whether to restore model weights from epoch with best metric. Default: True
    """
    
    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        mode='max',
        verbose=True,
        restore_best_weights=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_metric_min = np.Inf if mode == 'min' else -np.Inf
        
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, val_metric, model=None, epoch=0):
        """
        Check if training should stop.
        
        Args:
            val_metric (float): Current validation metric value
            model (nn.Module): Model to save best weights from
            epoch (int): Current epoch number
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.mode == 'max':
            score = val_metric
        else:
            score = -val_metric
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            self.val_metric_min = val_metric
            if model is not None:
                self.save_checkpoint(val_metric, model)
            return False
        
        # Check if there's improvement
        if self.mode == 'max':
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta
        
        if improved:
            if self.verbose:
                print(f'✅ Validation metric improved ({self.val_metric_min:.6f} → {val_metric:.6f}). '
                      f'Resetting patience counter.')
            self.best_score = score
            self.best_epoch = epoch
            self.val_metric_min = val_metric
            self.counter = 0
            if model is not None:
                self.save_checkpoint(val_metric, model)
        else:
            self.counter += 1
            if self.verbose:
                print(f'⚠️  No improvement in validation metric. '
                      f'Patience counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'🛑 Early stopping triggered! No improvement for {self.patience} validations.')
                    print(f'   Best metric: {self.val_metric_min:.6f} at epoch {self.best_epoch}')
                self.early_stop = True
                return True
        
        return False
    
    def save_checkpoint(self, val_metric, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            if self.val_metric_min != (np.Inf if self.mode == 'min' else -np.Inf):
                print(f'💾 Saving best model checkpoint (metric: {val_metric:.6f})')
        self.val_metric_min = val_metric
        # Store best model weights internally
        self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    def restore_best(self, model):
        """Restore best model weights if configured."""
        if self.restore_best_weights and hasattr(self, 'best_model_state'):
            if self.verbose:
                print(f'🔄 Restoring best model weights from epoch {self.best_epoch}')
            model.load_state_dict(self.best_model_state)
            return True
        return False


class MetricTracker:
    """
    Tracks training and validation metrics over time.
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.val_per_class_ious = []
        self.val_pixel_accs = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch, train_loss=None, val_loss=None, val_miou=None, 
               val_per_class_iou=None, val_pixel_acc=None, lr=None):
        """Update tracked metrics."""
        self.epochs.append(epoch)
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_miou is not None:
            self.val_mious.append(val_miou)
        if val_per_class_iou is not None:
            self.val_per_class_ious.append(val_per_class_iou)
        if val_pixel_acc is not None:
            self.val_pixel_accs.append(val_pixel_acc)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def get_summary(self):
        """Get summary of tracked metrics."""
        summary = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
            'val_per_class_ious': self.val_per_class_ious,
            'val_pixel_accs': self.val_pixel_accs,
            'learning_rates': self.learning_rates,
        }
        
        if self.val_mious:
            summary['best_val_miou'] = max(self.val_mious)
            summary['best_val_miou_epoch'] = self.epochs[self.val_mious.index(max(self.val_mious))]
        
        return summary
    
    def save(self, filepath):
        """Save metrics to file."""
        summary = self.get_summary()
        torch.save(summary, filepath)
        print(f"📊 Metrics saved to {filepath}")
    
    def load(self, filepath):
        """Load metrics from file."""
        summary = torch.load(filepath)
        self.epochs = summary.get('epochs', [])
        self.train_losses = summary.get('train_losses', [])
        self.val_losses = summary.get('val_losses', [])
        self.val_mious = summary.get('val_mious', [])
        self.val_per_class_ious = summary.get('val_per_class_ious', [])
        self.val_pixel_accs = summary.get('val_pixel_accs', [])
        self.learning_rates = summary.get('learning_rates', [])
        print(f"📊 Metrics loaded from {filepath}")


def check_overfitting(train_loss, val_loss, threshold=0.1):
    """
    Check if model is overfitting based on train/validation loss gap.
    
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        threshold (float): Threshold for overfitting detection. If val_loss > train_loss * (1 + threshold),
                          model is considered to be overfitting.
    
    Returns:
        bool: True if overfitting detected, False otherwise
    """
    if train_loss == 0:
        return False
    
    gap_ratio = (val_loss - train_loss) / train_loss
    return gap_ratio > threshold
