# FarSeg Generic Dataset Framework

This framework allows you to easily train and evaluate both FarSeg and FarSeg++ models on any segmentation dataset following a standardized directory structure. You only need to change the dataset name and configuration parameters - the rest of the process is automated.

## Environment Setup

Before using this framework, ensure you have the correct environment set up:

### Quick Setup (Recommended)

```bash
# Create the farsegpp environment with all dependencies
conda env create -f farsegpp_environment.yml
conda activate farsegpp
```

### Verified Environment Specifications
- **Python**: 3.9.23
- **PyTorch**: 2.4.0 with CUDA 11.8 support  
- **Ever Framework**: 0.5.2 (required for FarSeg++)
- **SimpleCV**: Latest from Z-Zheng GitHub repository
- **GPU Support**: Multi-GPU training ready
- **Mixed Precision**: Native PyTorch AMP available

For manual installation instructions, see the main [README.md](README.md).

## Dataset Directory Structure

Your datasets should follow this standardized structure:

```
/path/to/your/datasets/
└── your_dataset_name/
    ├── train/
    │   ├── dsm/        # Digital Surface Models (ignored for pure segmentation)
    │   ├── rgb/        # RGB input images (main input)
    │   ├── sar/        # Synthetic Aperture Radar (ignored for pure segmentation)
    │   └── sem/        # Semantic segmentation labels (uint8 format)
    ├── valid/
    │   ├── dsm/
    │   ├── rgb/
    │   ├── sar/
    │   └── sem/
    └── test/
        ├── dsm/
        ├── rgb/
        ├── sar/
        └── sem/
```

### Supported File Formats

The framework automatically detects and supports multiple image formats:
- **PNG** (.png, .PNG)
- **TIFF** (.tif, .tiff, .TIF, .TIFF) - including LZW compressed TIFF files
- **JPEG** (.jpg, .jpeg, .JPG, .JPEG)
- **BMP** (.bmp, .BMP)
- **GIF** (.gif, .GIF)

Both uppercase and lowercase extensions are supported. The framework will automatically find images regardless of the format used.

## Quick Start

### Option 1: Automated Pipeline (Recommended)

Use our convenient shell script that runs the complete pipeline:

1. **Configure the pipeline**: Edit `run.sh` and set your parameters:
   ```bash
   # Essential parameters to modify:
   DATASET_NAME="your_dataset_name"
   DATASET_PATH="/path/to/your/datasets"  
   NUM_CLASSES=5
   CLASS_VALUES="0,1,2,3,4"
   GPU_IDS="0,1"
   ```

2. **Execute the complete pipeline**:
   ```bash
   bash run.sh
   ```

The script will automatically handle all steps: analysis → configuration → validation → training → evaluation.

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

### Option 2: Manual Step-by-Step

### 1. Analyze Your Dataset

First, analyze your dataset to understand the class distribution:

```bash
# Analyze the training split (auto-detects file formats)
python analyze_dataset.py \
    --dataset_path /path/to/your/datasets \
    --dataset_name your_dataset_name \
    --split train \
    --output_dir ./analysis \
    --save_json \
    --plot

# For specific file extensions
python analyze_dataset.py \
    --dataset_path /path/to/your/datasets \
    --dataset_name your_dataset_name \
    --split train \
    --mask_extension .tif \
    --output_dir ./analysis \
    --save_json \
    --plot
```

This will:
- Count pixels for each class
- Show class distribution statistics
- Detect class imbalance issues
- Generate visualization plots
- Provide configuration recommendations

### 2. Generate Configuration

Based on the analysis results, generate a configuration file:

```bash
# Basic configuration
python generate_config.py \\
    --dataset_name your_dataset_name \\
    --num_classes 5 \\
    --data_root /path/to/your/datasets

# Advanced configuration with custom parameters
python generate_config.py \\
    --dataset_name your_dataset_name \\
    --num_classes 5 \\
    --data_root /path/to/your/datasets \\
    --class_values "0,1,2,3,4" \\
    --patch_size 896 \\
    --stride 512 \\
    --batch_size_train 4 \\
    --base_lr 0.007 \\
    --max_iters 60000
```

This creates:
- `configs/your_dataset_name/farseg_your_dataset_name.py` (Python config)
- `configs/your_dataset_name/config_your_dataset_name.json` (JSON reference)

### 3. Validate Configuration

Before training, validate your setup:

```bash
python train_generic.py \\
    --config configs/your_dataset_name/farseg_your_dataset_name.py \\
    --model_dir ./models/your_dataset_name \\
    --validate_config
```

This will:
- Check configuration file syntax
- Verify all data paths exist
- Count files in each directory
- Report any issues

### 4. Train the Model

Start training:

```bash
# Single GPU training
python train_generic.py \\
    --config configs/your_dataset_name/farseg_your_dataset_name.py \\
    --model_dir ./models/your_dataset_name

# Multi-GPU training (distributed)
python -m torch.distributed.launch --nproc_per_node=2 train_generic.py \\
    --config configs/your_dataset_name/farseg_your_dataset_name.py \\
    --model_dir ./models/your_dataset_name
```

### 5. Evaluate the Model

After training, evaluate the model:

```bash
python train_generic.py \\
    --config configs/your_dataset_name/farseg_your_dataset_name.py \\
    --model_dir ./models/your_dataset_name \\
    --eval_only
```

## File Descriptions

### Core Framework Files

- **`data/generic_dataset.py`**: Generic dataset loader that works with the standardized directory structure
- **`generate_config.py`**: Automatically generates configuration files for any dataset
- **`analyze_dataset.py`**: Analyzes dataset class distribution and provides recommendations
- **`train_generic.py`**: Unified training and evaluation script

### Generated Files

- **`configs/{dataset_name}/farseg_{dataset_name}.py`**: Python configuration file for training
- **`configs/{dataset_name}/config_{dataset_name}.json`**: JSON configuration for reference
- **`analysis/{dataset_name}_analysis.json`**: Detailed dataset analysis results
- **`analysis/class_distribution.png`**: Class distribution visualization

## Key Features

### 🔧 **Automatic Configuration Generation**
- Just specify dataset name, number of classes, and data path
- Automatically configures paths, transforms, and model parameters
- Generates both Python and JSON configuration files

### 📊 **Dataset Analysis**
- Analyzes class distribution in semantic masks
- Detects class imbalance issues
- Provides configuration recommendations
- Generates visualization plots

### 🎯 **Modality Flexibility**
- Uses only RGB images for segmentation (ignores DSM and SAR)
- Works with uint8 semantic mask format
- Supports any number of classes

### ⚙️ **Easy Customization**
- Override any parameter via command line
- Support for different patch sizes, batch sizes, learning rates
- Flexible class value mapping

## Example Workflows

### Small Dataset (< 1000 images)
```bash
# Use smaller patches and higher learning rate
python generate_config.py \\
    --dataset_name small_dataset \\
    --num_classes 3 \\
    --data_root /data \\
    --patch_size 512 \\
    --stride 256 \\
    --batch_size_train 8 \\
    --base_lr 0.01 \\
    --max_iters 30000
```

### Large Dataset (> 10000 images)
```bash
# Use larger patches and more iterations
python generate_config.py \\
    --dataset_name large_dataset \\
    --num_classes 10 \\
    --data_root /data \\
    --patch_size 1024 \\
    --stride 768 \\
    --batch_size_train 2 \\
    --base_lr 0.005 \\
    --max_iters 100000
```

### High-Resolution Dataset
```bash
# Use very large patches for high-res images
python generate_config.py \\
    --dataset_name highres_dataset \\
    --num_classes 5 \\
    --data_root /data \\
    --patch_size 1536 \\
    --stride 1024 \\
    --batch_size_train 1 \\
    --base_lr 0.003
```

## Troubleshooting

### Common Issues

1. **"No mask files found"**
   - Check that your masks are in the `sem/` directory
   - Verify file extensions (default is `.png`)
   - Use `--mask_extension` parameter if different

2. **"Class imbalance detected"**
   - The framework will automatically use focal loss
   - Consider data augmentation for minority classes
   - Check if background class dominates (normal for segmentation)

3. **"CUDA out of memory"**
   - Reduce batch size: `--batch_size_train 1`
   - Reduce patch size: `--patch_size 512`
   - Use gradient checkpointing

4. **"Config validation failed"**
   - Check that all data directories exist
   - Verify RGB and semantic mask files have matching names
   - Ensure mask files are readable

### Performance Tips

- **Use SSD storage** for faster data loading
- **Optimize num_workers** (typically 4-8 per GPU)
- **Use mixed precision** training (default O1 level)
- **Monitor GPU utilization** and adjust batch size accordingly

## Advanced Usage

### Custom Transforms

You can modify the generated config file to add custom transforms:

```python
# In your generated config file
transforms=[
    GenericRemoveColorMap(class_values=[0,1,2], num_classes=3),
    segm.RandomHorizontalFlip(0.5),
    segm.RandomVerticalFlip(0.5),
    segm.RandomRotate90K((0, 1, 2, 3)),
    segm.RandomBrightness(0.1),  # Add custom transform
    segm.RandomContrast(0.1),    # Add custom transform
    segm.FixedPad((896, 896), 255),
    segm.ToTensor(True),
    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
]
```

### Multi-Scale Training

Modify the config for multi-scale training:

```python
# Multiple patch sizes
patch_configs = [
    dict(patch_size=512, stride=256),
    dict(patch_size=768, stride=384), 
    dict(patch_size=1024, stride=512)
]
```

### Class Weighting

For imbalanced datasets, add class weights:

```python
# In model.params.loss
loss=dict(
    cls_weight=1.0,
    ignore_index=255,
    class_weights=[1.0, 2.0, 3.0, 1.5]  # Weight for each class
)
```

## Citations

If you use this framework, please cite the original FarSeg paper:

```bibtex
@inproceedings{zheng2020foreground,
  title={Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4096--4105},
  year={2020}
}
```
