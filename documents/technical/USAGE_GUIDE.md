# FarSeg Pipeline Usage Guide

## Quick Start with run.sh

The `run.sh` script provides a complete automated pipeline for training FarSeg/FarSeg++ models on any dataset.

### 1. Edit Configuration

Open `run.sh` and modify the parameters in the configuration section:

```bash
# Essential parameters to modify:
DATASET_NAME="your_dataset_name"              # Name of your dataset
DATASET_PATH="/path/to/your/datasets"          # Root path to datasets directory  
NUM_CLASSES=5                                   # Number of classes in your dataset
CLASS_VALUES="0,1,2,3,4"                      # Comma-separated class values
GPU_IDS="0,1"                                  # GPU IDs to use
```

### 2. Execute Pipeline

```bash
# Run the complete pipeline
bash run.sh
```

The script will automatically:
1. ✅ Analyze your dataset
2. ✅ Generate configuration files
3. ✅ Validate the setup
4. ✅ Train the model
5. ✅ Evaluate the results

### 3. Pipeline Control

You can control which steps to run by modifying these flags in `run.sh`:

```bash
RUN_ANALYSIS=true                             # Run dataset analysis
RUN_CONFIG_GEN=true                          # Generate configuration files
RUN_VALIDATION=true                          # Validate configuration before training
RUN_TRAINING=true                            # Run model training
RUN_EVALUATION=true                          # Run model evaluation after training
```

## Example Configurations

### Small Dataset (< 1000 images)
```bash
PATCH_SIZE=512
STRIDE=256
BATCH_SIZE_TRAIN=8
BASE_LR=0.01
MAX_ITERS=30000
```

### Large Dataset (> 10000 images)
```bash
PATCH_SIZE=1024
STRIDE=768
BATCH_SIZE_TRAIN=2
BASE_LR=0.005
MAX_ITERS=100000
```

### High-Resolution Dataset
```bash
PATCH_SIZE=1536
STRIDE=1024
BATCH_SIZE_TRAIN=1
BASE_LR=0.003
```

## Manual Step-by-Step Execution

If you prefer to run steps individually:

### Step 1: Dataset Analysis
```bash
conda activate farsegpp
python analyze_dataset.py \
    --dataset_path /path/to/datasets \
    --dataset_name your_dataset \
    --split train \
    --output_dir ./analysis \
    --save_json --plot
```

### Step 2: Generate Configuration
```bash
python generate_config.py \
    --dataset_name your_dataset \
    --num_classes 5 \
    --data_root /path/to/datasets \
    --patch_size 896 \
    --stride 512
```

### Step 3: Train Model
```bash
# Single GPU
python train_generic.py \
    --config configs/your_dataset/farseg_your_dataset.py \
    --model_dir ./models/your_dataset

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_generic.py \
    --config configs/your_dataset/farseg_your_dataset.py \
    --model_dir ./models/your_dataset
```

### Step 4: Evaluate Model
```bash
python train_generic.py \
    --config configs/your_dataset/farseg_your_dataset.py \
    --model_dir ./models/your_dataset \
    --eval_only
```

## Environment Requirements

Make sure you have the correct environment set up:

```bash
# Create from our environment file
conda env create -f farsegpp_environment.yml
conda activate farsegpp
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `BATCH_SIZE_TRAIN` and/or `PATCH_SIZE`
2. **No mask files found**: Check `DATASET_PATH` and directory structure
3. **Class imbalance**: The framework automatically handles this with focal loss
4. **Multi-GPU issues**: Ensure `GPU_IDS` are available and `CUDA_VISIBLE_DEVICES` is set

### Directory Structure Expected

```
/path/to/your/datasets/
└── your_dataset_name/
    ├── train/
    │   ├── rgb/        # RGB input images
    │   └── sem/        # Semantic segmentation labels
    ├── valid/
    │   ├── rgb/
    │   └── sem/
    └── test/
        ├── rgb/
        └── sem/
```

### Performance Tips

- Use SSD storage for faster data loading
- Optimize `NUM_WORKERS` (typically 4-8 per GPU)
- Monitor GPU utilization and adjust batch size accordingly
- Use larger patch sizes for high-resolution images
