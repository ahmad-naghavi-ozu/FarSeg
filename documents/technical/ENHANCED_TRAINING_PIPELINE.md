# Enhanced FarSeg/FarSeg++ Training Pipeline with Validation Strategy

## Branch: `feature/patch-size-optimization-and-isaid-cleanup-validation`

This document describes the enhanced training pipeline with validation-based early stopping, building upon the `feature/patch-size-optimization-and-isaid-cleanup` branch.

---

## Overview of Current Pipeline (Before Enhancement)

### 1. **Architecture**

The codebase supports two model architectures:

- **FarSeg** (`module/farseg.py`): ResNet backbone + FPN + Scene Relations
- **FarSegPP** (`module/farsegpp.py`): Enhanced with MiT/SegFormer backbones, PPM, FSR modules

### 2. **Current Training Strategy** (Iteration-Based)

**Key characteristics:**
- Fixed iteration count (`max_iters` = 60,000 or 120,000)
- Polynomial learning rate decay from `base_lr` to ~0
- **No validation during training** (`eval_per_epoch=False`, `eval_interval_epoch=999`)
- **No early stopping** mechanism
- **No patience counter** for convergence
- Evaluation happens **post-training** only

**Training Configuration:**
```python
train = dict(
    num_iters=60000,                # Fixed iterations
    eval_per_epoch=False,           # No validation
    eval_interval_epoch=999,        # Effectively disabled
    save_ckpt_interval_epoch=999,   # No frequent checkpoints
    log_interval_step=50,           # Logging only
)

learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.007,
        power=0.9,
        max_iters=60000,            # LR decays to 0 at max_iters
    )
)
```

### 3. **Dataset Handling**

**Patch-Based Training:**
- Optimized for 512×512 images
- Patch size: 256×256
- Stride: 128 (50% overlap)
- Generates 9 patches per 512×512 image

**Data Splits:**
- `train/`: Training set
- `valid/`: Validation set (currently **unused during training**)
- `test/`: Test set (used only for final evaluation)

**Current Strategy:**
- Fusion mode: Combines `train/` + `valid/` for training (no validation split)
- Maximizes training data but loses validation-based monitoring

### 4. **Loss Functions**

**FarSegPP Multi-Task Loss:**
```python
loss = {
    "objectness": {
        "log_objectness_iou_sigmoid": {...},
        "dice": {...},
        "bce": {...},
    },
    "semantic": {
        "annealing_softmax_focal": {...},
        "log_objectness_iou": {...},
    }
}
```

### 5. **Current Workflow**

```bash
# Step 1: Generate config (with train+valid fusion)
python generate_config.py --use_train_valid_fusion

# Step 2: Train (no validation)
python train_simple.py --config config.py --model_dir ./models

# Step 3: Evaluate on test set (post-training)
python eval_simple.py --config config.py --model_dir ./models
```

---

## Limitations of Current Approach

### **Problems:**

1. ❌ **No convergence monitoring**: Can't detect when model stops improving
2. ❌ **Wasted computation**: May train beyond optimal point
3. ❌ **No overfitting detection**: No validation loss to compare against training loss
4. ❌ **Fixed iteration count**: Requires manual tuning for different datasets
5. ❌ **Suboptimal LR schedule**: Polynomial decay assumes training completes at `max_iters`
6. ❌ **No best model selection**: Final model may not be the best performing

### **Why This Matters:**

- **For small datasets**: Model may overfit before reaching `max_iters`
- **For large datasets**: Model may converge before `max_iters`, wasting GPU time
- **For research**: Hard to compare models fairly without validation-based stopping

---

## Enhanced Pipeline with Validation Strategy

### **Key Enhancements:**

1. ✅ **Validation during training**: Periodic evaluation on validation set
2. ✅ **Early stopping with patience**: Stop when validation mIoU plateaus
3. ✅ **Best model checkpoint**: Save model with highest validation mIoU
4. ✅ **Adaptive LR scheduling**: ReduceLROnPlateau based on validation metrics
5. ✅ **Overfitting detection**: Monitor train vs. validation loss divergence
6. ✅ **Configurable validation split**: Separate train/valid sets

### **New Training Configuration:**

```python
train = dict(
    # Validation settings
    eval_per_epoch=True,              # Enable validation
    eval_interval_epoch=1,            # Validate every N epochs
    eval_interval_steps=500,          # Or validate every N steps
    
    # Early stopping
    early_stopping=dict(
        enabled=True,
        patience=10,                  # Stop if no improvement for 10 validations
        min_delta=0.001,              # Minimum improvement threshold (mIoU)
        metric='val_mIoU',            # Metric to monitor
        mode='max',                   # Maximize mIoU
    ),
    
    # Checkpointing
    save_best_only=True,              # Save only when validation improves
    save_checkpoint_interval=5,       # Also save periodic checkpoints
    
    # Max iterations (safety limit)
    max_iters=60000,                  # Upper bound (may stop earlier)
    max_epochs=None,                  # Optional epoch limit
    
    # Logging
    log_interval_step=50,
    log_validation_metrics=True,
)

learning_rate = dict(
    type='plateau',                   # New adaptive scheduler
    params=dict(
        base_lr=0.007,
        mode='max',                   # Maximize mIoU
        factor=0.5,                   # Reduce LR by 50%
        patience=5,                   # Reduce after 5 validations without improvement
        min_lr=1e-6,                  # Minimum learning rate
        threshold=0.001,              # Improvement threshold
    )
)

# Alternative: Keep polynomial but with early stopping
learning_rate = dict(
    type='poly_with_warmup',
    params=dict(
        base_lr=0.007,
        power=0.9,
        max_iters=60000,              # Used for schedule, but training may stop early
        warmup_iters=1000,            # Gradual warmup
    )
)
```

### **Data Configuration:**

```python
data = dict(
    train=dict(
        type="GenericSegmentationDataLoader",
        params=dict(
            image_dir="/path/to/train/rgb",
            mask_dir="/path/to/train/sem",
            # ... other params
        )
    ),
    valid=dict(                       # NEW: Separate validation set
        type="GenericSegmentationDataLoader",
        params=dict(
            image_dir="/path/to/valid/rgb",
            mask_dir="/path/to/valid/sem",
            # ... similar to train
        )
    ),
    test=dict(
        # ... existing test config
    )
)
```

---

## Implementation Details

### **New Files Created:**

1. **`train_with_validation.py`**: Enhanced training script with validation loop
2. **`module/lr_scheduler.py`**: Custom learning rate schedulers
3. **`module/early_stopping.py`**: Early stopping callback implementation
4. **`module/metrics.py`**: Validation metrics computation
5. **`generate_config_validation.py`**: Config generator for validation-based training

### **Modified Files:**

1. **`generate_config.py`**: Added `--use_validation_strategy` flag
2. **`run.sh`**: Added validation mode support
3. **Configuration templates**: Updated with validation parameters

### **Training Workflow (New):**

```bash
# Step 1: Generate config with validation strategy
python generate_config.py \
    --dataset_name DFC2023S \
    --use_validation_strategy \
    --validation_split 0.2 \
    --early_stopping_patience 10

# Step 2: Train with validation
python train_with_validation.py \
    --config configs/farsegpp/DFC2023S/farseg_DFC2023S_validation.py \
    --model_dir ./models/farsegpp/DFC2023S \
    --gpu_ids 0

# Training will:
# - Validate every epoch (or every N steps)
# - Save best model based on validation mIoU
# - Stop early if no improvement for N epochs
# - Adjust learning rate when validation plateaus

# Step 3: Evaluate best model on test set
python eval_simple.py \
    --config configs/farsegpp/DFC2023S/farseg_DFC2023S_validation.py \
    --ckpt_path ./models/farsegpp/DFC2023S/best_model.pth \
    --output_dir ./predictions/farsegpp/DFC2023S
```

---

## Validation Metrics

### **Computed During Training:**

- **mIoU (mean Intersection over Union)**: Primary metric for early stopping
- **Per-class IoU**: Monitor individual class performance
- **Pixel Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision/recall
- **Training Loss**: Monitor for overfitting (compare with validation loss)

### **Logged Information:**

```
Epoch 15/∞ | Step 7500/60000
Train Loss: 0.2345 | Val Loss: 0.2876
Train mIoU: 0.8234 | Val mIoU: 0.7892
Learning Rate: 0.00350
Best Val mIoU: 0.7954 (Epoch 12)
Patience: 3/10
```

---

## Early Stopping Logic

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
            return False
        
        if self.mode == 'max':
            improved = (val_metric - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - val_metric) > self.min_delta
        
        if improved:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop
```

---

## Learning Rate Scheduling Options

### **Option 1: ReduceLROnPlateau (Recommended)**

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',         # Maximize validation mIoU
    factor=0.5,         # Multiply LR by 0.5
    patience=5,         # Wait 5 validations
    min_lr=1e-6
)

# Usage in training loop:
scheduler.step(val_miou)
```

**Advantages:**
- Adapts to actual convergence behavior
- Works well with early stopping
- No need to specify max_iters
- Can recover from plateaus

### **Option 2: Polynomial with Early Stopping**

```python
# Keep existing polynomial decay but allow early stopping
def poly_lr_with_early_stop(optimizer, current_iter, max_iters, base_lr, power=0.9):
    lr = base_lr * (1 - current_iter / max_iters) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Advantages:**
- Compatible with existing configs
- Predictable decay schedule
- Can still stop early when converged

### **Option 3: CosineAnnealingWarmRestarts**

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,           # Restart every 1000 steps
    T_mult=2,           # Double restart period each time
    eta_min=1e-6
)
```

**Advantages:**
- Periodic restarts help escape local minima
- Good for long training
- Works well with early stopping

---

## Checkpoint Management

### **Checkpoint Types:**

1. **Best Model**: `best_model.pth` (highest validation mIoU)
2. **Latest Model**: `latest_model.pth` (most recent checkpoint)
3. **Periodic Checkpoints**: `model-{step}.pth` (every N steps)
4. **Early Stop Checkpoint**: `early_stop_model.pth` (saved when training stops)

### **Checkpoint Contents:**

```python
checkpoint = {
    'epoch': epoch,
    'global_step': global_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_miou': best_val_miou,
    'train_loss_history': train_losses,
    'val_loss_history': val_losses,
    'val_miou_history': val_mious,
    'early_stopping_counter': early_stopping.counter,
    'config': config,
}
```

---

## Comparison: Old vs. New Pipeline

| Aspect | Old Pipeline | New Pipeline |
|--------|--------------|--------------|
| **Validation** | None during training | Every epoch/N steps |
| **Stopping Criterion** | Fixed iterations | Early stopping + max_iters |
| **LR Schedule** | Polynomial decay | ReduceLROnPlateau (adaptive) |
| **Best Model Selection** | Final checkpoint | Highest validation mIoU |
| **Overfitting Detection** | None | Train vs. validation loss |
| **Training Time** | Always full max_iters | Stops when converged |
| **GPU Efficiency** | May waste compute | Stops at optimal point |
| **Model Quality** | May miss best checkpoint | Guaranteed best validation |
| **Hyperparameter Tuning** | Manual max_iters tuning | Automatic via patience |

---

## Usage Examples

### **Example 1: Quick Training with Validation**

```bash
./run.sh \
    --action train \
    --dataset DFC2023S \
    --model_type farsegpp \
    --use_validation \
    --early_stopping_patience 10 \
    --max_iters 60000
```

### **Example 2: Custom Validation Configuration**

```bash
python train_with_validation.py \
    --config configs/farsegpp/DFC2023S/farseg_DFC2023S.py \
    --model_dir ./models/farsegpp/DFC2023S \
    --validation_interval_epochs 1 \
    --early_stopping_patience 15 \
    --early_stopping_min_delta 0.0005 \
    --lr_scheduler plateau \
    --lr_patience 5 \
    --gpu_ids 0
```

### **Example 3: Resume Training with Validation**

```bash
python train_with_validation.py \
    --config configs/farsegpp/DFC2023S/farseg_DFC2023S.py \
    --model_dir ./models/farsegpp/DFC2023S \
    --resume_from ./models/farsegpp/DFC2023S/latest_model.pth \
    --gpu_ids 0
```

---

## Benefits of Enhanced Pipeline

### **For Research:**

1. ✅ **Fair model comparison**: All models use same stopping criterion
2. ✅ **Reproducibility**: Convergence-based stopping is more reproducible than fixed iterations
3. ✅ **Hyperparameter insights**: Learning curves show when models converge
4. ✅ **Overfitting analysis**: Can detect and quantify overfitting

### **For Production:**

1. ✅ **Resource efficiency**: Don't waste GPU time on converged models
2. ✅ **Best model guaranteed**: Always get the best performing checkpoint
3. ✅ **Robustness**: Early stopping prevents overfitting on small datasets
4. ✅ **Flexibility**: Works across different dataset sizes

### **For Development:**

1. ✅ **Faster iteration**: Quicker feedback on model changes
2. ✅ **Better monitoring**: Rich logging of training dynamics
3. ✅ **Easier debugging**: Validation metrics expose issues earlier
4. ✅ **Confidence**: Know when model has truly converged

---

## Backward Compatibility

The enhanced pipeline maintains **full backward compatibility**:

- Old configs still work with `train_simple.py` (iteration-based training)
- New configs work with `train_with_validation.py` (validation-based training)
- Can mix and match: use validation but disable early stopping
- `run.sh` supports both modes via `--use_validation` flag

---

## Future Enhancements

Potential additions to the validation pipeline:

1. **Cross-validation**: K-fold CV for small datasets
2. **Multi-metric early stopping**: Consider multiple metrics (mIoU + F1)
3. **Gradient accumulation**: Support larger effective batch sizes
4. **Mixed precision**: Further optimize memory and speed
5. **Distributed validation**: Parallel validation on multiple GPUs
6. **Hyperparameter search**: Auto-tune LR, patience, etc.
7. **Active learning**: Select hard examples for annotation based on validation errors

---

## Summary

This enhanced pipeline transforms FarSeg/FarSeg++ from a fixed-iteration training system into an **adaptive, validation-driven training framework**. Key improvements:

- **Smarter training**: Stop when model converges, not at arbitrary iteration count
- **Better models**: Automatically select best checkpoint via validation
- **Resource efficient**: Save GPU time by detecting convergence
- **More robust**: Prevent overfitting with early stopping
- **Research-ready**: Proper validation enables fair model comparisons

The validation strategy makes the training process more intelligent, efficient, and aligned with modern deep learning best practices.
