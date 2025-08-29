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
MODEL_TYPE="farsegpp"                            # Model type: "farseg" or "farsegpp"
BACKBONE="resnet50"                            # Backbone: "resnet50", "mit_b2", etc.

# Training Parameters
PATCH_SIZE=896                                 # Input patch size
STRIDE=512                                     # Patch stride for training
BATCH_SIZE_TRAIN=8                            # Training batch size (reduced for single GPU)
BATCH_SIZE_VAL=1                              # Validation batch size
BASE_LR=0.007                                  # Base learning rate
MAX_ITERS=60000                               # Maximum training iterations

# Hardware Configuration
GPU_IDS="0"                                    # GPU ID to use (train_simple.py supports single GPU only)
NUM_WORKERS=4                                  # Number of data loading workers

# File Extensions (auto-detection if not specified)
IMAGE_EXTENSION=""                             # Leave empty for auto-detection (.png, .tif, .jpg)
MASK_EXTENSION=""                             # Leave empty for auto-detection

# Output Directories
MODEL_DIR="./models"                          # Directory to save trained models
ANALYSIS_DIR="./analysis"                     # Directory to save analysis results
CONFIG_DIR="./configs"                        # Directory for generated configs
PREDICTIONS_DIR="./predictions"               # Directory to save prediction outputs

# Pipeline Control Flags
RUN_DATASET_ANALYSIS=true                    # Run dataset analysis to understand data distribution
RUN_CONFIG_GEN=true                          # Generate configuration files
USE_TRAIN_VALID_FUSION=true                  # Use train+valid fusion for training (manuscript strategy)
RUN_TRAINING=true                            # Run model training
RUN_EVALUATION=true                          # Run model evaluation after training
FORCE_PREDICTIONS=true                       # Force regeneration of predictions even if they exist
RESUME_TRAINING=true                        # Resume training from latest checkpoint if available

#===============================================================================
# ADVANCED PARAMETERS (usually don't need to change)
#===============================================================================

# Environment
CONDA_ENV="farsegpp"                          # Conda environment name

# Analysis Parameters
SAVE_ANALYSIS_JSON=true                        # Save analysis results as JSON (fixed JSON serialization issue)
GENERATE_PLOTS=true                          # Generate visualization plots

# Training Parameters
MIXED_PRECISION=true                         # Use mixed precision training
SAVE_FREQUENCY=5000                          # Model saving frequency (iterations)
VAL_FREQUENCY=2000                           # Validation frequency (iterations)

#===============================================================================
# PIPELINE EXECUTION - DO NOT MODIFY BELOW THIS LINE
#===============================================================================

# Parse command line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --max_iters)
            MAX_ITERS="$2"
            shift 2
            ;;
        --batch_size_train)
            BATCH_SIZE_TRAIN="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --force_predictions)
            FORCE_PREDICTIONS=true
            shift 1
            ;;
        --use_train_valid_fusion)
            USE_TRAIN_VALID_FUSION=true
            shift 1
            ;;
        --resume)
            RESUME_TRAINING=true
            shift 1
            ;;
        --action)
            ACTION="$2"
            case $ACTION in
                all)
                    RUN_DATASET_ANALYSIS=true
                    RUN_CONFIG_GEN=true
                    USE_TRAIN_VALID_FUSION=true
                    RUN_TRAINING=true
                    RUN_EVALUATION=true
                    ;;
                analysis)
                    RUN_DATASET_ANALYSIS=true
                    RUN_CONFIG_GEN=false
                    USE_TRAIN_VALID_FUSION=false
                    RUN_TRAINING=false
                    RUN_EVALUATION=false
                    ;;
                config)
                    RUN_DATASET_ANALYSIS=false
                    RUN_CONFIG_GEN=true
                    USE_TRAIN_VALID_FUSION=true
                    RUN_TRAINING=false
                    RUN_EVALUATION=false
                    ;;
                train)
                    RUN_DATASET_ANALYSIS=false
                    RUN_CONFIG_GEN=false
                    USE_TRAIN_VALID_FUSION=true
                    RUN_TRAINING=true
                    RUN_EVALUATION=false
                    ;;
                eval)
                    RUN_DATASET_ANALYSIS=false
                    RUN_CONFIG_GEN=false
                    USE_TRAIN_VALID_FUSION=false
                    RUN_TRAINING=false
                    RUN_EVALUATION=true
                    ;;
                prepare)
                    RUN_DATASET_ANALYSIS=true
                    RUN_CONFIG_GEN=true
                    USE_TRAIN_VALID_FUSION=true
                    RUN_TRAINING=false
                    RUN_EVALUATION=false
                    ;;
                *)
                    echo "Unknown action: $ACTION"
                    echo "Valid actions: all, analysis, config, train, eval, prepare"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        --help|-h)
            echo "FarSeg/FarSeg++ Training Pipeline"
            echo ""
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET_NAME      Name of the dataset (default: DFC2023S)"
            echo "  --max_iters VALUE          Maximum training iterations (default: 60000)"
            echo "  --batch_size_train VALUE   Training batch size (default: 2)"
            echo "  --gpu_ids VALUE            GPU IDs to use (default: 0,1)"
            echo "  --force_predictions        Force regeneration of predictions even if they exist"
            echo "  --use_train_valid_fusion   Combine train and valid sets for training (no validation split)"
            echo "  --resume                   Resume training from latest checkpoint if available"
            echo "  --action ACTION            Pipeline action to perform (default: all)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Actions:"
            echo "  all       - Run complete pipeline (analysis + config + train + eval)"
            echo "  analysis  - Run dataset analysis only" 
            echo "  config    - Generate configuration files only"
            echo "  train     - Run model training only (with train+valid fusion)"
            echo "  eval      - Run model evaluation only (on test split)"
            echo "  prepare   - Run preparation steps (analysis + config)"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --action all --dataset DFC2023mini --max_iters 60"
            echo "  ./run.sh --action train --dataset DFC2023S --gpu_ids 0,1" 
            echo "  ./run.sh --action eval --force_predictions"
            echo "  ./run.sh --action prepare --use_train_valid_fusion"
            echo "  ./run.sh --action train --resume  # Resume from latest checkpoint"
            echo "  ./run.sh --action train --dataset DFC2023S --resume  # Resume training with dataset specification"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run.sh [OPTIONS]"
            echo "Use --help for detailed information about available options and actions."
            exit 1
            ;;
    esac
done

# Adjust save and validation frequencies based on max_iters
if [ "$MAX_ITERS" -le 100 ]; then
    SAVE_FREQUENCY=10  # Save every 10 iterations for debugging
    VAL_FREQUENCY=5    # Validate every 5 iterations
elif [ "$MAX_ITERS" -le 1000 ]; then
    SAVE_FREQUENCY=100
    VAL_FREQUENCY=50
else
    SAVE_FREQUENCY=5000
    VAL_FREQUENCY=2000
fi

echo "============================================================================="
echo "FarSeg/FarSeg++ Training Pipeline"
echo "============================================================================="
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_TYPE"
echo "Classes: $NUM_CLASSES"
echo "GPU(s): $GPU_IDS"
echo "Max Iterations: $MAX_ITERS"
echo "Save Frequency: $SAVE_FREQUENCY"
echo "Val Frequency: $VAL_FREQUENCY"
echo "Train+Valid Fusion: $USE_TRAIN_VALID_FUSION"
echo "Force Predictions: $FORCE_PREDICTIONS"
echo "Resume Training: $RESUME_TRAINING"
echo ""
echo "Pipeline Actions:"
echo "  Dataset Analysis: $RUN_DATASET_ANALYSIS"
echo "  Config Generation: $RUN_CONFIG_GEN"  
echo "  Training: $RUN_TRAINING"
echo "  Evaluation: $RUN_EVALUATION"
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
mkdir -p $PREDICTIONS_DIR

#===============================================================================
# STEP 1: Dataset Analysis
#===============================================================================

if [ "$RUN_DATASET_ANALYSIS" = true ]; then
    echo ""
    echo "Step 1: Analyzing dataset..."
    echo "============================================================================="
    
    ANALYSIS_CMD="python analyze_dataset.py \
        --dataset_path $DATASET_PATH \
        --dataset_name $DATASET_NAME \
        --split train \
        --output_dir $ANALYSIS_DIR/$DATASET_NAME"
    
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
    
    if [ "$USE_TRAIN_VALID_FUSION" = true ]; then
        CONFIG_CMD="$CONFIG_CMD --use_train_valid_fusion"
    fi
    
    echo "Running: $CONFIG_CMD"
    eval $CONFIG_CMD
    
    echo "✅ Configuration files generated!"
fi

#===============================================================================
# STEP 3: Model Training
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
    
    # Add resume flag if specified
    if [ "$RESUME_TRAINING" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --resume latest"
        echo "🔄 Resuming training from latest checkpoint"
    else
        echo "🆕 Starting fresh training"
    fi
    
    echo "Running: $TRAIN_CMD"
    eval $TRAIN_CMD
    
    echo "✅ Model training completed!"
fi

#===============================================================================
# STEP 4: Model Evaluation
#===============================================================================

if [ "$RUN_EVALUATION" = true ]; then
    echo ""
    echo "Step 5: Evaluating model..."
    echo "============================================================================="
    
    CONFIG_FILE="$CONFIG_DIR/$DATASET_NAME/farseg_$DATASET_NAME.py"
    MODEL_OUTPUT_DIR="$MODEL_DIR/$DATASET_NAME"
    EVAL_OUTPUT_DIR="$MODEL_OUTPUT_DIR/evaluation"
    
    # Use the new generic evaluation script
    EVAL_CMD="python -u eval_simple.py \
        --config $CONFIG_FILE \
        --model_dir $MODEL_OUTPUT_DIR \
        --output_dir $EVAL_OUTPUT_DIR \
        --gpu_ids $GPU_IDS"
    
    # Add force predictions flag if specified
    if [ "$FORCE_PREDICTIONS" = true ]; then
        EVAL_CMD="$EVAL_CMD --force_predictions"
    fi
    
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
echo "  - Analysis: $ANALYSIS_DIR/$DATASET_NAME"
echo "  - Configs: $CONFIG_DIR/$DATASET_NAME"
echo "  - Predictions: $MODEL_DIR/$DATASET_NAME/evaluation/full_predictions"
echo "============================================================================="

# Display final model performance (if available)
if [ -f "$MODEL_DIR/$DATASET_NAME/evaluation/eval_results.txt" ]; then
    echo ""
    echo "Final Model Performance:"
    echo "============================================================================="
    cat "$MODEL_DIR/$DATASET_NAME/evaluation/eval_results.txt"
elif [ -f "$MODEL_DIR/$DATASET_NAME/eval_results.txt" ]; then
    echo ""
    echo "Final Model Performance:"
    echo "============================================================================="
    cat "$MODEL_DIR/$DATASET_NAME/eval_results.txt"
fi

echo ""
echo "To use the trained model for evaluation/inference, activate the environment and use:"
echo "  conda activate $CONDA_ENV"
echo "  python eval_simple.py --config $CONFIG_FILE --model_dir $MODEL_OUTPUT_DIR --output_dir $MODEL_OUTPUT_DIR/inference"
echo ""
echo "Or for quick evaluation with the existing configuration:"
echo "  ./run.sh --action eval --dataset $DATASET_NAME"
echo ""
