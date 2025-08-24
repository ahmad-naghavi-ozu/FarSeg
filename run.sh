#!/bin/bash

#===============================================================================
# FarSeg/FarSeg++ Training Pipeline Script
# This script provides a convenient way to run the complete FarSeg pipeline
# Just set the parameters below and execute: bash run.sh
#===============================================================================

set -e  # Exit on any error

#===============================================================================
# CONFIGURATION PARAMETERS - MODIFY THESE FOR YOUR DATASET
#===============================================================================

# Dataset Configuration
DATASET_NAME="DFC2023S"              # Name of your dataset
DATASET_PATH="/home/asfand/Ahmad/datasets/"          # Root path to datasets directory
NUM_CLASSES=2                                   # Number of classes in your dataset
CLASS_VALUES="0,1"                      # Comma-separated class values

# Model Configuration  
MODEL_TYPE="farseg"                            # Model type: "farseg" or "farsegpp"
BACKBONE="resnet50"                            # Backbone: "resnet50", "mit_b2", etc.

# Training Parameters
PATCH_SIZE=896                                 # Input patch size
STRIDE=512                                     # Patch stride for training
BATCH_SIZE_TRAIN=2                            # Training batch size (reduced for single GPU)
BATCH_SIZE_VAL=1                              # Validation batch size
BASE_LR=0.007                                  # Base learning rate
MAX_ITERS=60000                               # Maximum training iterations

# Hardware Configuration
GPU_IDS="2"                                  # GPU IDs to use (switching to GPU 3 to avoid memory conflicts)
NUM_WORKERS=4                                  # Number of data loading workers

# File Extensions (auto-detection if not specified)
IMAGE_EXTENSION=""                             # Leave empty for auto-detection (.png, .tif, .jpg)
MASK_EXTENSION=""                             # Leave empty for auto-detection

# Output Directories
MODEL_DIR="./models"                          # Directory to save trained models
ANALYSIS_DIR="./analysis"                     # Directory to save analysis results
CONFIG_DIR="./configs"                        # Directory for generated configs

# Pipeline Control Flags
RUN_ANALYSIS=false                             # Run dataset analysis (already completed)
RUN_CONFIG_GEN=false                          # Generate configuration files (already completed)
RUN_VALIDATION=false                          # Validate configuration before training (disabled for train_simple.py)
RUN_TRAINING=true                            # Run model training
RUN_EVALUATION=true                          # Run model evaluation after training

#===============================================================================
# ADVANCED PARAMETERS (usually don't need to change)
#===============================================================================

# Environment
CONDA_ENV="farsegpp"                          # Conda environment name

# Analysis Parameters
SAVE_ANALYSIS_JSON=false                       # Save analysis results as JSON (disabled due to JSON serialization issue)
GENERATE_PLOTS=true                          # Generate visualization plots

# Training Parameters
MIXED_PRECISION=true                         # Use mixed precision training
SAVE_FREQUENCY=5000                          # Model saving frequency (iterations)
VAL_FREQUENCY=2000                           # Validation frequency (iterations)

#===============================================================================
# PIPELINE EXECUTION - DO NOT MODIFY BELOW THIS LINE
#===============================================================================

echo "============================================================================="
echo "FarSeg/FarSeg++ Training Pipeline"
echo "============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_TYPE"
echo "Classes: $NUM_CLASSES"
echo "GPU(s): $GPU_IDS"
echo "============================================================================="

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Verify environment
echo "Verifying environment..."
python -c "
import torch
import ever
import simplecv
print(f'✅ Python: {__import__(\"sys\").version.split()[0]}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPUs: {torch.cuda.device_count()}')
print(f'✅ Ever: {ever.__version__}')
"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Create output directories
mkdir -p $MODEL_DIR
mkdir -p $ANALYSIS_DIR
mkdir -p $CONFIG_DIR

#===============================================================================
# STEP 1: Dataset Analysis
#===============================================================================

if [ "$RUN_ANALYSIS" = true ]; then
    echo ""
    echo "Step 1: Analyzing dataset..."
    echo "============================================================================="
    
    ANALYSIS_CMD="python analyze_dataset.py \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASET_NAME \
        --split train \
        --output_dir $ANALYSIS_DIR"
    
    if [ "$SAVE_ANALYSIS_JSON" = true ]; then
        ANALYSIS_CMD="$ANALYSIS_CMD --save_json"
    fi
    
    if [ "$GENERATE_PLOTS" = true ]; then
        ANALYSIS_CMD="$ANALYSIS_CMD --plot"
    fi
    
    if [ -n "$MASK_EXTENSION" ]; then
        ANALYSIS_CMD="$ANALYSIS_CMD --mask_extension $MASK_EXTENSION"
    fi
    
    echo "Running: $ANALYSIS_CMD"
    eval $ANALYSIS_CMD
    
    echo "✅ Dataset analysis completed!"
fi

#===============================================================================
# STEP 2: Configuration Generation
#===============================================================================

if [ "$RUN_CONFIG_GEN" = true ]; then
    echo ""
    echo "Step 2: Generating configuration files..."
    echo "============================================================================="
    
    CONFIG_CMD="python generate_config.py \
        --dataset_name $DATASET_NAME \
        --num_classes $NUM_CLASSES \
        --data_root $DATASET_PATH \
        --patch_size $PATCH_SIZE \
        --stride $STRIDE \
        --batch_size_train $BATCH_SIZE_TRAIN \
        --batch_size_test $BATCH_SIZE_VAL \
        --base_lr $BASE_LR \
        --max_iters $MAX_ITERS"
    
    if [ -n "$CLASS_VALUES" ]; then
        CONFIG_CMD="$CONFIG_CMD --class_values $CLASS_VALUES"
    fi
    
    echo "Running: $CONFIG_CMD"
    eval $CONFIG_CMD
    
    echo "✅ Configuration files generated!"
fi

#===============================================================================
# STEP 3: Configuration Validation
#===============================================================================

if [ "$RUN_VALIDATION" = true ]; then
    echo ""
    echo "Step 3: Validating configuration..."
    echo "============================================================================="
    
    CONFIG_FILE="$CONFIG_DIR/$DATASET_NAME/farseg_$DATASET_NAME.py"
    MODEL_OUTPUT_DIR="$MODEL_DIR/$DATASET_NAME"
    
    VALIDATION_CMD="python train_simple.py \
        --config $CONFIG_FILE \
        --model_dir $MODEL_OUTPUT_DIR \
        --validate_config"
    
    echo "Running: $VALIDATION_CMD"
    eval $VALIDATION_CMD
    
    echo "✅ Configuration validation completed!"
fi

#===============================================================================
# STEP 4: Model Training
#===============================================================================

if [ "$RUN_TRAINING" = true ]; then
    echo ""
    echo "Step 4: Training model..."
    echo "============================================================================="
    
    CONFIG_FILE="$CONFIG_DIR/$DATASET_NAME/farseg_$DATASET_NAME.py"
    MODEL_OUTPUT_DIR="$MODEL_DIR/$DATASET_NAME"
    
    # Use simplified training (train_simple.py doesn't support distributed training yet)
    echo "Training with train_simple.py (optimized for stability)..."
    TRAIN_CMD="CUDA_VISIBLE_DEVICES=$GPU_IDS python train_simple.py \
        --config $CONFIG_FILE \
        --model_dir $MODEL_OUTPUT_DIR"
    
    echo "Running: $TRAIN_CMD"
    eval $TRAIN_CMD
    
    echo "✅ Model training completed!"
fi

#===============================================================================
# STEP 5: Model Evaluation
#===============================================================================

if [ "$RUN_EVALUATION" = true ]; then
    echo ""
    echo "Step 5: Evaluating model..."
    echo "============================================================================="
    
    CONFIG_FILE="$CONFIG_DIR/$DATASET_NAME/farseg_$DATASET_NAME.py"
    MODEL_OUTPUT_DIR="$MODEL_DIR/$DATASET_NAME"
    
    EVAL_CMD="python train_simple.py \
        --config $CONFIG_FILE \
        --model_dir $MODEL_OUTPUT_DIR \
        --eval_only"
    
    echo "Running: $EVAL_CMD"
    eval $EVAL_CMD
    
    echo "✅ Model evaluation completed!"
fi

#===============================================================================
# PIPELINE COMPLETED
#===============================================================================

echo ""
echo "============================================================================="
echo "🎉 FarSeg Pipeline Completed Successfully!"
echo "============================================================================="
echo "Results saved in:"
echo "  - Models: $MODEL_DIR/$DATASET_NAME"
echo "  - Analysis: $ANALYSIS_DIR"
echo "  - Configs: $CONFIG_DIR/$DATASET_NAME"
echo "============================================================================="

# Display final model performance (if available)
if [ -f "$MODEL_OUTPUT_DIR/eval_results.txt" ]; then
    echo ""
    echo "Final Model Performance:"
    echo "============================================================================="
    cat "$MODEL_OUTPUT_DIR/eval_results.txt"
fi

echo ""
echo "To use the trained model for inference, activate the environment and use:"
echo "  conda activate $CONDA_ENV"
echo "  python inference_script.py --config $CONFIG_FILE --model_dir $MODEL_OUTPUT_DIR"
echo ""
