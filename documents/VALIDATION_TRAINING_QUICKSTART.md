# Quick Start Guide: Validation-Based Training

This guide shows how to use the new validation-based training strategy in FarSeg/FarSeg++.

## Prerequisites

Ensure your dataset follows the standardized structure:
```
/path/to/datasets/YourDataset/
├── train/
│   ├── rgb/      # Training images
│   └── sem/      # Training masks
├── valid/
│   ├── rgb/      # Validation images
│   └── sem/      # Validation masks
└── test/
    ├── rgb/      # Test images
    └── sem/      # Test masks
```

## Method 1: Using run.sh Script (Recommended)

The easiest way is to modify `run.sh` and use the validation mode:

```bash
# Edit run.sh and set:
USE_TRAIN_VALID_FUSION=false    # Use separate train/valid splits
RUN_CONFIG_GEN=true             # Generate config with validation
RUN_TRAINING=true               # Run training
RUN_EVALUATION=true             # Run evaluation

# Then run:
./run.sh --action all --dataset DFC2023S --model_type farsegpp
```

## Method 2: Manual Step-by-Step

### Step 1: Generate Configuration with Validation Support

```bash
python generate_config.py \
    --dataset_name DFC2023S \
    --num_classes 2 \
    --class_values "0,1" \
    --data_root /home/asfand/Ahmad/datasets/ \
    --patch_size 256 \
    --stride 128 \
    --batch_size_train 100 \
    --batch_size_test 1 \
    --base_lr 0.007 \
    --max_iters 60000 \
    --model_type farsegpp \
    --use_validation \
    --lr_scheduler plateau \
    --output_dir ./configs/farsegpp/DFC2023S
```

**Key flags:**
- `--use_validation`: Adds validation split configuration
- `--lr_scheduler plateau`: Use ReduceLROnPlateau (adaptive to validation)
- Do NOT use `--use_train_valid_fusion` (keep train/valid separate)

### Step 2: Train with Validation Strategy

```bash
python train_with_validation.py \
    --config ./configs/farsegpp/DFC2023S/farseg_DFC2023S.py \
    --model_dir ./models/farsegpp/DFC2023S \
    --gpu_ids 0 \
    --validation_interval_epochs 1 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.001 \
    --lr_scheduler plateau \
    --max_iters 60000
```

**Key parameters:**
- `--validation_interval_epochs 1`: Validate after every epoch
- `--validation_interval_steps 500`: Or validate every 500 steps (optional)
- `--early_stopping_patience 10`: Stop if no improvement for 10 validations
- `--early_stopping_min_delta 0.001`: Minimum mIoU improvement threshold
- `--lr_scheduler plateau`: Use adaptive LR (or 'poly', 'poly_warmup', 'cosine_warmup')

### Step 3: Monitor Training

The training will output:

```
Epoch 15/∞ | Step 7500/60000
Train Loss: 0.2345 | Val Loss: 0.2876
Train mIoU: N/A (computed on full pass)
Val mIoU: 0.7892
Learning Rate: 0.00350
Best Val mIoU: 0.7954 (Epoch 12)
Patience: 3/10

✅ Validation metric improved (0.7892 → 0.7954). Resetting patience counter.
💾 Saving best model checkpoint (metric: 0.7954)
```

Training will automatically:
- Validate every epoch (or every N steps)
- Save best model based on validation mIoU
- Reduce learning rate when validation plateaus
- Stop early when no improvement for N validations

### Step 4: Evaluate Best Model

```bash
python eval_simple.py \
    --config ./configs/farsegpp/DFC2023S/farseg_DFC2023S.py \
    --ckpt_path ./models/farsegpp/DFC2023S/best_model.pth \
    --output_dir ./predictions/farsegpp/DFC2023S
```

## Understanding the Checkpoints

After training, you'll find several checkpoints:

```
./models/farsegpp/DFC2023S/
├── best_model.pth              # Model with highest validation mIoU ⭐
├── latest_model.pth            # Most recent checkpoint
├── early_stop_model.pth        # Saved when early stopping triggered
├── checkpoint_epoch_5.pth      # Periodic checkpoints every 5 epochs
├── checkpoint_epoch_10.pth
├── training_metrics.pth        # Full metric history (PyTorch format)
└── training_metrics.json       # Full metric history (human-readable)
```

**Use `best_model.pth` for final evaluation!**

## Resume Training

If training was interrupted, resume from the latest checkpoint:

```bash
python train_with_validation.py \
    --config ./configs/farsegpp/DFC2023S/farseg_DFC2023S.py \
    --model_dir ./models/farsegpp/DFC2023S \
    --resume_from ./models/farsegpp/DFC2023S/latest_model.pth \
    --gpu_ids 0
```

## Comparing Strategies

### Old Strategy (Iteration-Based)
```bash
# Generate config with train+valid fusion
python generate_config.py \
    --dataset_name DFC2023S \
    --use_train_valid_fusion \
    --max_iters 60000 \
    ...

# Train for fixed iterations
python train_simple.py \
    --config config.py \
    --model_dir ./models \
    --max_iters 60000
```

**Characteristics:**
- ✅ Simple and predictable
- ✅ Works well for benchmarking
- ❌ May overfit or undertrain
- ❌ Wastes computation if model converges early
- ❌ No automatic best model selection

### New Strategy (Validation-Based)
```bash
# Generate config with validation split
python generate_config.py \
    --dataset_name DFC2023S \
    --use_validation \
    --lr_scheduler plateau \
    ...

# Train with validation and early stopping
python train_with_validation.py \
    --config config.py \
    --model_dir ./models \
    --validation_interval_epochs 1 \
    --early_stopping_patience 10
```

**Characteristics:**
- ✅ Stops automatically when converged
- ✅ Guarantees best model selection
- ✅ Adaptive learning rate
- ✅ Detects overfitting
- ✅ More efficient use of GPU time
- ❌ Requires separate validation set
- ❌ Slightly more complex setup

## Configuration Options

### Learning Rate Schedulers

**1. Plateau (Recommended for validation-based training)**
```bash
--lr_scheduler plateau
```
- Reduces LR when validation mIoU plateaus
- Adaptive to actual convergence
- Works best with early stopping

**2. Polynomial (Traditional)**
```bash
--lr_scheduler poly
```
- Decays from base_lr to 0 over max_iters
- Predictable schedule
- Can still use early stopping

**3. Polynomial with Warmup**
```bash
--lr_scheduler poly_warmup
```
- Linear warmup for 1000 steps
- Then polynomial decay
- Good for stability

**4. Cosine Annealing with Warmup**
```bash
--lr_scheduler cosine_warmup
```
- Smooth cosine decay
- Optional restarts
- Good for long training

### Validation Intervals

**By Epoch (Simpler)**
```bash
--validation_interval_epochs 1    # Validate every epoch
--validation_interval_epochs 2    # Validate every 2 epochs
```

**By Steps (More frequent)**
```bash
--validation_interval_steps 500   # Validate every 500 steps
--validation_interval_steps 1000  # Validate every 1000 steps
```

### Early Stopping Tuning

**Conservative (Allow more training)**
```bash
--early_stopping_patience 20      # Wait 20 validations
--early_stopping_min_delta 0.0001 # Very small improvement counts
```

**Aggressive (Stop quickly)**
```bash
--early_stopping_patience 5       # Wait only 5 validations
--early_stopping_min_delta 0.005  # Require substantial improvement
```

## Tips and Best Practices

1. **Start with validation-based training** if you have a separate validation set
2. **Use plateau scheduler** with early stopping for best results
3. **Validate every epoch** for datasets with <1000 training samples
4. **Validate every N steps** for larger datasets (e.g., every 500-1000 steps)
5. **Set patience = 10-15** as a good starting point
6. **Monitor training curves** in `training_metrics.json`
7. **Always evaluate using `best_model.pth`**, not final checkpoint
8. **Keep max_iters as safety limit** (e.g., 60000) but expect early stopping

## Troubleshooting

**Problem: Training stops too early**
- Increase `--early_stopping_patience` (try 15-20)
- Decrease `--early_stopping_min_delta` (try 0.0001)
- Check if validation set is too small/noisy

**Problem: Training never stops**
- Decrease patience (try 5-10)
- Check if model is actually improving (look at metrics)
- Verify validation set is representative

**Problem: Validation mIoU oscillates**
- Reduce learning rate (`--base_lr 0.003`)
- Use warmup: `--lr_scheduler poly_warmup`
- Increase batch size if possible
- Check for data issues (e.g., class imbalance)

**Problem: Out of memory during validation**
- Reduce validation batch size in config
- Reduce validation frequency: `--validation_interval_epochs 2`
- Use gradient checkpointing (model-specific)

## Example Workflows

### Small Dataset (< 500 samples)
```bash
python train_with_validation.py \
    --config config.py \
    --model_dir ./models \
    --validation_interval_epochs 1 \
    --early_stopping_patience 15 \
    --lr_scheduler plateau \
    --base_lr 0.005 \
    --max_iters 30000
```

### Medium Dataset (500-2000 samples)
```bash
python train_with_validation.py \
    --config config.py \
    --model_dir ./models \
    --validation_interval_epochs 1 \
    --early_stopping_patience 10 \
    --lr_scheduler plateau \
    --max_iters 60000
```

### Large Dataset (> 2000 samples)
```bash
python train_with_validation.py \
    --config config.py \
    --model_dir ./models \
    --validation_interval_steps 1000 \
    --early_stopping_patience 10 \
    --lr_scheduler poly_warmup \
    --max_iters 120000
```

## Viewing Results

Check training metrics:
```bash
# View metrics JSON
cat ./models/farsegpp/DFC2023S/training_metrics.json

# Quick summary
python -c "
import json
with open('./models/farsegpp/DFC2023S/training_metrics.json') as f:
    m = json.load(f)
    print(f'Best mIoU: {m[\"best_val_miou\"]:.4f} at epoch {m[\"best_val_miou_epoch\"]}')
    print(f'Total epochs: {len(m[\"epochs\"])}')
"
```

Compare with test set performance:
```bash
# Evaluate best model on test set
python eval_simple.py \
    --config config.py \
    --ckpt_path ./models/farsegpp/DFC2023S/best_model.pth \
    --output_dir ./predictions

# Results will be in:
# ./predictions/eval_results.txt
```

---

**For more details, see:**
- `documents/technical/ENHANCED_TRAINING_PIPELINE.md` - Complete technical documentation
- `documents/technical/OPTIMIZATION_SUMMARY.md` - Patch size optimization details
- `module/early_stopping.py` - Early stopping implementation
- `module/metrics.py` - Validation metrics computation
- `module/lr_scheduler.py` - Learning rate scheduler options
