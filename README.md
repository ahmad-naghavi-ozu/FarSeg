<h2 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h2>
<!-- <h5 align="center">Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery</h5> -->



<h5><a href="http://zhuozheng.top/">Zhuo Zheng</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a> and Ailong Ma</h5>


<div align="center">
  <img src="https://raw.githubusercontent.com/Z-Zheng/images_repo/master/farseg.png"><br><br>
</div>

This is an official implementation of FarSeg in our CVPR 2020 paper [Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.pdf).

---------------------
## News
- 2024/03, source code of FarSeg++ is released.
- 2023/10, [UV6K dataset](https://zenodo.org/record/8404754) is publcily available.
- 2023/07, FarSeg++ is accepted by IEEE TPAMI.

## Citation
If you use FarSeg or FarSeg++ in your research, please cite the following paper:
```text
@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}
@article{zheng2023farseg++,
  title={FarSeg++: Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong and Zhang, Liangpei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  volume={45},
  number={11},
  pages={13715-13729},
  publisher={IEEE}
}
```

## Getting Started

### Environment Setup

We provide a complete conda environment configuration for both FarSeg and FarSeg++. The environment has been tested and verified to work with all components.

#### Quick Setup (Recommended)

Create the environment from our pre-configured file:

```bash
# Create the farsegpp environment with all dependencies
conda env create -f farsegpp_environment.yml

# Activate the environment
conda activate farsegpp
```

#### Manual Setup

If you prefer manual installation:

```bash
# Create environment
conda create -n farsegpp python=3.9 -y
conda activate farsegpp

# Install PyTorch with CUDA support
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install scientific packages
mamba install scipy scikit-image opencv pandas matplotlib seaborn tqdm tensorboardx albumentations -y

# Install timm for transformer models
pip install timm

# Install SimpleCV from GitHub (required for FarSeg)
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git

# Install Ever framework (required for FarSeg++)
pip install ever-beta

# Fix protobuf compatibility for SimpleCV
pip install "protobuf<3.21"
```

#### Verified Environment Specifications:
- **Python**: 3.9.23
- **PyTorch**: 2.4.0 with CUDA 11.8 support
- **Ever Framework**: 0.5.2 (for FarSeg++)
- **SimpleCV**: Latest from Z-Zheng GitHub repository
- **GPU Support**: 4x NVIDIA GeForce RTX 2080 Ti tested
- **Mixed Precision**: Native PyTorch AMP available

#### Requirements:
- CUDA-compatible GPU (recommended)
- Python >= 3.9 (for FarSeg++)
- Python >= 3.6 (for original FarSeg only)

### Prepare iSAID Dataset

```bash
ln -s </path/to/iSAID> ./isaid_segm
```

### Evaluate Model
#### 1. download pretrained weight in this [link](https://github.com/Z-Zheng/FarSeg/releases/download/v1.0/farseg50.pth)

#### 2. move weight file to log directory
```bash
mkdir -vp ./log/isaid_segm/farseg50
mv ./farseg50.pth ./log/isaid_segm/farseg50/model-60000.pth
```
#### 3. inference on iSAID val
```bash
bash ./scripts/eval_farseg50.sh
```

### Train Model
```bash
bash ./scripts/train_farseg50.sh
```

## 🚀 Quick Start with Generic Framework

For training on custom datasets, use our automated pipeline:

```bash
# 1. Edit configuration in run.sh
# 2. Execute the complete pipeline
bash run.sh
```

See [README_GENERIC.md](README_GENERIC.md) for detailed instructions on training with custom datasets.

## ⚡ Recent Pipeline Improvements (2025)

### Key Features Added
- **Action-Based Control**: `--action all|train|eval|prepare` for fine-grained pipeline control
- **Resume Training**: Simple `--resume` flag for continuing interrupted training
- **Train+Valid Fusion**: Manuscript-compliant training strategy combining train and validation splits
- **Progress Tracking**: Real-time progress bars with fixed output buffering
- **GPU Optimization**: Single GPU training with memory management and batch size recommendations

### Hardware Configuration
- **GPU Support**: Single GPU training (train_simple.py limitation)
- **Multi-GPU**: Not supported in current implementation (planned for future)
- **Memory Management**: Optimized batch sizes for stable single GPU training
- **Recommended**: RTX 3080+ (12GB+ VRAM) for optimal performance

### Usage Examples
```bash
# Complete pipeline with resume capability
./run.sh --action all --dataset DFC2023S --resume --batch_size_train 8

# Training only with train+valid fusion
./run.sh --action train --dataset DFC2023S --use_train_valid_fusion

# Evaluation only with force predictions
./run.sh --action eval --dataset DFC2023S --force_predictions
```

## 📚 Documentation

### Core Documentation
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Complete usage guide for the automated pipeline
- **[README_GENERIC.md](README_GENERIC.md)**: Generic dataset training framework
- **[ENVIRONMENT.md](ENVIRONMENT.md)**: Complete environment setup and troubleshooting

### Technical Documentation (2025)
- **[documents/technical/PIPELINE_IMPROVEMENTS.md](documents/technical/PIPELINE_IMPROVEMENTS.md)**: Comprehensive pipeline improvements summary
- **[documents/technical/RESUME_TRAINING.md](documents/technical/RESUME_TRAINING.md)**: Resume training functionality guide
- **[documents/technical/GPU_CONFIGURATION.md](documents/technical/GPU_CONFIGURATION.md)**: GPU setup and optimization guide

### Environment Files
- `farsegpp_environment.yml`: Complete conda environment specification
- `farsegpp_conda_packages.txt`: Conda package list
- `farsegpp_pip_packages.txt`: Pip package list

## 🔧 Environment Management

The repository includes complete environment specifications:

```bash
# Quick setup from environment file
conda env create -f farsegpp_environment.yml
conda activate farsegpp

# Verify installation
python -c "import torch, ever, simplecv; print('✅ Environment ready')"
```


