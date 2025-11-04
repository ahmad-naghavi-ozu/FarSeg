#!/bin/bash
#===============================================================================
# Test Script for Validation-Based Training on DFC2023mini
# This script tests the new validation strategy on a small dataset
#===============================================================================

set -e  # Exit on any error

echo "============================================================================="
echo "Testing Validation-Based Training Pipeline on DFC2023mini"
echo "============================================================================="

# Test Configuration
DATASET_NAME="DFC2023mini"
DATA_ROOT="/home/asfand/Ahmad/datasets/DFC2023mini"
MODEL_TYPE="farsegpp"
NUM_CLASSES=2
GPU_IDS="0"

# Validation parameters
LR_SCHEDULER="plateau"
VALIDATION_INTERVAL_EPOCHS=1
EARLY_STOPPING_PATIENCE=5  # Small patience for quick testing
EARLY_STOPPING_MIN_DELTA=0.001

# Output directories
CONFIG_DIR="./configs/${MODEL_TYPE}/${DATASET_NAME}"
MODEL_DIR="./models/${MODEL_TYPE}/${DATASET_NAME}"
LOGS_DIR="./logs"

# Activate environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate farsegpp

# Create directories
mkdir -p "${CONFIG_DIR}"
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOGS_DIR}"

# Step 1: Verify dataset structure
echo ""
echo "Step 1: Verifying dataset structure..."
echo "============================================================================="

if [ ! -d "${DATASET_PATH}/${DATASET_NAME}" ]; then
    echo "❌ Dataset directory not found: ${DATASET_PATH}/${DATASET_NAME}"
    exit 1
fi

echo "Checking dataset splits..."
for SPLIT in train valid test; do
    RGB_DIR="${DATASET_PATH}/${DATASET_NAME}/${SPLIT}/rgb"
    SEM_DIR="${DATASET_PATH}/${DATASET_NAME}/${SPLIT}/sem"
    
    if [ -d "$RGB_DIR" ] && [ -d "$SEM_DIR" ]; then
        RGB_COUNT=$(ls -1 "$RGB_DIR" | wc -l)
        SEM_COUNT=$(ls -1 "$SEM_DIR" | wc -l)
        echo "  ✅ $SPLIT split: $RGB_COUNT images, $SEM_COUNT masks"
    else
        echo "  ❌ $SPLIT split: Missing directories"
        echo "     RGB: $RGB_DIR"
        echo "     SEM: $SEM_DIR"
        exit 1
    fi
done

# Step 2: Generate configuration with validation support
echo ""
echo "Step 2: Generating configuration with validation support..."
echo "============================================================================="

CONFIG_FILE="${CONFIG_DIR}/farseg_${DATASET_NAME}.py"

python generate_config.py \
    --dataset_name "$DATASET_NAME" \
    --num_classes "$NUM_CLASSES" \
    --class_values "$CLASS_VALUES" \
    --data_root "$DATASET_PATH" \
    --patch_size "$PATCH_SIZE" \
    --stride "$STRIDE" \
    --batch_size_train "$BATCH_SIZE_TRAIN" \
    --batch_size_test "$BATCH_SIZE_VAL" \
    --base_lr "$BASE_LR" \
    --max_iters "$MAX_ITERS" \
    --model_type "$MODEL_TYPE" \
    --use_validation \
    --lr_scheduler "$LR_SCHEDULER" \
    --output_dir "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    echo "✅ Configuration generated: $CONFIG_FILE"
else
    echo "❌ Configuration generation failed!"
    exit 1
fi

# Step 3: Verify configuration
echo ""
echo "Step 3: Verifying configuration..."
echo "============================================================================="

echo "Checking if validation split is configured..."
if grep -q "'valid'" "$CONFIG_FILE" || grep -q '"valid"' "$CONFIG_FILE"; then
    echo "  ✅ Validation split found in configuration"
else
    echo "  ❌ Validation split NOT found in configuration!"
    exit 1
fi

echo "Checking LR scheduler configuration..."
if grep -q "plateau" "$CONFIG_FILE"; then
    echo "  ✅ Plateau LR scheduler configured"
else
    echo "  ⚠️  Plateau scheduler not found, may be using default"
fi

# Step 4: Run validation-based training
echo ""
echo "Step 4: Running validation-based training..."
echo "============================================================================="

LOG_FILE="${LOGS_DIR}/test_validation_${DATASET_NAME}_$(date '+%Y%m%d_%H%M%S').log"

export CUDA_VISIBLE_DEVICES=$GPU_IDS

TRAIN_CMD="python train_with_validation.py \
    --config $CONFIG_FILE \
    --model_dir $MODEL_DIR \
    --validation_interval_epochs $VALIDATION_INTERVAL_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --early_stopping_min_delta $EARLY_STOPPING_MIN_DELTA \
    --lr_scheduler $LR_SCHEDULER \
    --max_iters $MAX_ITERS \
    --gpu_ids 0"

echo "Training command:"
echo "$TRAIN_CMD"
echo ""
echo "Starting training... (logs: $LOG_FILE)"
echo ""

if eval $TRAIN_CMD 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "❌ Training failed! Check log: $LOG_FILE"
    exit 1
fi

# Step 5: Verify outputs
echo ""
echo "Step 5: Verifying training outputs..."
echo "============================================================================="

echo "Checking for saved checkpoints..."
if [ -f "${MODEL_DIR}/best_model.pth" ]; then
    echo "  ✅ Best model checkpoint found"
else
    echo "  ❌ Best model checkpoint NOT found"
fi

if [ -f "${MODEL_DIR}/latest_model.pth" ]; then
    echo "  ✅ Latest model checkpoint found"
else
    echo "  ⚠️  Latest model checkpoint NOT found"
fi

echo "Checking for training metrics..."
if [ -f "${MODEL_DIR}/training_metrics.json" ]; then
    echo "  ✅ Training metrics (JSON) found"
    echo ""
    echo "  Metrics summary:"
    python -c "
import json
with open('${MODEL_DIR}/training_metrics.json') as f:
    m = json.load(f)
    if 'best_val_miou' in m:
        print(f'    Best validation mIoU: {m[\"best_val_miou\"]:.4f} at epoch {m[\"best_val_miou_epoch\"]}')
    print(f'    Total epochs: {len(m[\"epochs\"])}')
    if 'val_mious' in m and len(m['val_mious']) > 0:
        print(f'    Validation mIoU history: {[round(x, 4) for x in m[\"val_mious\"]]}')
"
else
    echo "  ⚠️  Training metrics NOT found"
fi

# Final summary
echo ""
echo "============================================================================="
echo "🎉 Validation Training Test Completed!"
echo "============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_TYPE"
echo "Config: $CONFIG_FILE"
echo "Model dir: $MODEL_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Check training curves in: ${MODEL_DIR}/training_metrics.json"
echo "  2. Evaluate best model:"
echo "     python eval_simple.py --config $CONFIG_FILE \\"
echo "       --ckpt_path ${MODEL_DIR}/best_model.pth \\"
echo "       --output_dir ./predictions/${MODEL_TYPE}/${DATASET_NAME}"
echo "============================================================================="
