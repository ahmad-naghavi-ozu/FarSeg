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

# Training parameters
CLASS_VALUES="0,1"
PATCH_SIZE=256
STRIDE=128
BATCH_SIZE_TRAIN=8
BATCH_SIZE_VAL=1
BASE_LR=0.007
MAX_ITERS=150

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

if [ ! -d "${DATA_ROOT}" ]; then
    echo "❌ Dataset directory not found: ${DATA_ROOT}"
    exit 1
fi

echo "Checking dataset splits..."
for SPLIT in train valid test; do
    RGB_DIR="${DATA_ROOT}/${SPLIT}/rgb"
    SEM_DIR="${DATA_ROOT}/${SPLIT}/sem"
    
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
    --data_root "$(dirname "$DATA_ROOT")" \
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

# Step 6: Run evaluation on test set
echo ""
echo "Step 6: Running evaluation on test set..."
echo "============================================================================="

PREDICTION_DIR="./predictions/${MODEL_TYPE}/${DATASET_NAME}"
EVAL_LOG="${LOGS_DIR}/test_evaluation_${DATASET_NAME}_$(date '+%Y%m%d_%H%M%S').log"

echo "Evaluating best model on test split..."
echo "Output directory: $PREDICTION_DIR"
echo ""

EVAL_CMD="python eval_simple.py \
    --config $CONFIG_FILE \
    --model_dir $MODEL_DIR \
    --output_dir $PREDICTION_DIR \
    --gpu_ids 0 \
    --force_predictions"

if eval $EVAL_CMD 2>&1 | tee "$EVAL_LOG"; then
    echo ""
    echo "✅ Evaluation completed successfully!"
else
    echo ""
    echo "❌ Evaluation failed! Check log: $EVAL_LOG"
    exit 1
fi

# Step 7: Verify evaluation outputs
echo ""
echo "Step 7: Verifying evaluation outputs..."
echo "============================================================================="

echo "Checking for prediction files..."
if [ -d "${PREDICTION_DIR}/predictions" ]; then
    PRED_COUNT=$(ls -1 "${PREDICTION_DIR}/predictions" | wc -l)
    echo "  ✅ Predictions directory found: $PRED_COUNT files"
else
    echo "  ❌ Predictions directory NOT found"
fi

if [ -f "${PREDICTION_DIR}/eval_results.json" ]; then
    echo "  ✅ Evaluation results (JSON) found"
elif [ -f "${PREDICTION_DIR}/eval_results.txt" ]; then
    echo "  ✅ Evaluation results (TXT) found"
else
    echo "  ❌ Evaluation results NOT found"
fi

echo "Checking for visualizations..."
if [ -f "${PREDICTION_DIR}/confusion_matrix.png" ]; then
    echo "  ✅ Confusion matrix visualization found"
else
    echo "  ⚠️  Confusion matrix NOT found"
fi

if [ -d "${PREDICTION_DIR}/prediction_samples" ]; then
    SAMPLE_COUNT=$(ls -1 "${PREDICTION_DIR}/prediction_samples" 2>/dev/null | wc -l)
    echo "  ✅ Prediction samples directory found: $SAMPLE_COUNT samples"
else
    echo "  ⚠️  Prediction samples NOT found"
fi

# Display evaluation metrics
echo ""
echo "Evaluation Metrics Summary:"
echo "============================================================================="
if [ -f "${PREDICTION_DIR}/eval_results.txt" ]; then
    cat "${PREDICTION_DIR}/eval_results.txt"
elif [ -f "${PREDICTION_DIR}/eval_results.json" ]; then
    python -c "
import json
with open('${PREDICTION_DIR}/eval_results.json') as f:
    results = json.load(f)
    print(f\"  Overall Accuracy: {results.get('overall_accuracy', 'N/A'):.4f}\")
    print(f\"  Mean IoU: {results.get('mean_iou', 'N/A'):.4f}\")
    if 'per_class_iou' in results:
        print(f\"  Per-Class IoU:\")
        for i, iou in enumerate(results['per_class_iou']):
            print(f\"    Class {i}: {iou:.4f}\")
"
fi

# Final summary
echo ""
echo "============================================================================="
echo "🎉 Validation Training Pipeline Test Completed!"
echo "============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_TYPE"
echo "Config: $CONFIG_FILE"
echo "Model dir: $MODEL_DIR"
echo "Predictions: $PREDICTION_DIR"
echo ""
echo "Logs:"
echo "  Training log: $LOG_FILE"
echo "  Evaluation log: $EVAL_LOG"
echo ""
echo "Generated files:"
echo "  ✓ Training metrics: ${MODEL_DIR}/training_metrics.json"
echo "  ✓ Best model: ${MODEL_DIR}/best_model.pth"
echo "  ✓ Predictions: ${PREDICTION_DIR}/predictions/"
echo "  ✓ Evaluation results: ${PREDICTION_DIR}/eval_results.*"
echo "  ✓ Visualizations: ${PREDICTION_DIR}/*.png"
echo "============================================================================="
