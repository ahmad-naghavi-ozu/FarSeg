# FarSeg++ Environment Documentation

This document provides complete information about the working FarSeg++ environment configuration.

## Environment Overview

- **Environment Name**: `farsegpp`
- **Python Version**: 3.9.23
- **Creation Date**: August 2025
- **Status**: ✅ Fully Functional - All components tested and working

## Key Framework Versions

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.9.23 | Base interpreter |
| **PyTorch** | 2.4.0 | Deep learning framework |
| **CUDA** | 11.8 | GPU acceleration |
| **Ever Framework** | 0.5.2 | Core FarSeg++ dependency |
| **SimpleCV** | Latest (GitHub) | Computer vision utilities |
| **timm** | Latest | Transformer models (MiT) |
| **Protobuf** | 3.20.3 | Downgraded for compatibility |

## Hardware Compatibility

- **GPU**: 4x NVIDIA GeForce RTX 2080 Ti (tested)
- **Memory**: 11GB VRAM per GPU
- **CUDA Compute Capability**: 7.5
- **Multi-GPU Training**: ✅ Supported

## Installation Methods

### Method 1: From Environment File (Recommended)

```bash
# Clone/download the farsegpp_environment.yml file
conda env create -f farsegpp_environment.yml
conda activate farsegpp
```

### Method 2: Manual Installation

```bash
# Create base environment
conda create -n farsegpp python=3.9 -y
conda activate farsegpp

# Core PyTorch installation
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Scientific computing stack
mamba install numpy scipy scikit-image scikit-learn -y
mamba install opencv pandas matplotlib seaborn -y
mamba install tqdm tensorboardx albumentations -y

# Deep learning utilities
pip install timm  # Transformer models

# FarSeg-specific packages
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
pip install ever-beta

# Compatibility fix
pip install "protobuf<3.21"
```

## Component Testing

All components have been thoroughly tested:

### ✅ Core Framework Tests
```bash
conda activate farsegpp
python -c "
import torch, ever, simplecv, sys
print('✅ Python:', sys.version.split()[0])
print('✅ PyTorch:', torch.__version__)  
print('✅ CUDA:', torch.version.cuda)
print('✅ Ever:', ever.__version__)
print('✅ GPUs:', torch.cuda.device_count())
"
```

### ✅ FarSeg++ Module Tests
```bash
python -c "
from module.farsegpp import FarSegPP
from module.mit import mit_b2
from module.comm import *
print('✅ All FarSeg++ modules imported successfully')
"
```

### ✅ Mixed Precision Support
```bash
python -c "
from torch.cuda.amp import autocast, GradScaler
print('✅ Native PyTorch AMP available')
"
```

## Critical Compatibility Notes

### 1. Python Version Requirements
- **FarSeg++**: Requires Python ≥ 3.8 (due to ever framework syntax)
- **Original FarSeg**: Compatible with Python ≥ 3.6
- **Recommended**: Python 3.9 for optimal compatibility

### 2. Protobuf Version
- **Issue**: SimpleCV requires protobuf < 3.21
- **Solution**: `pip install "protobuf<3.21"` (downgrades from 6.31.1)

### 3. Ever Framework
- **Repository**: `ever-beta` from PyPI
- **Version**: 0.5.2
- **Dependency**: Required for FarSeg++ and MiT transformers

### 4. SimpleCV Source
- **Correct Source**: GitHub Z-Zheng/SimpleCV (PyTorch-based)
- **Avoid**: PyPI SimpleCV v1.3 (2012 OpenCV-based version)

## Environment Files

The following files are provided for environment management:

1. **`farsegpp_environment.yml`**: Complete conda environment export
2. **`farsegpp_conda_packages.txt`**: Conda package list with versions
3. **`farsegpp_pip_packages.txt`**: Pip package list with versions

## Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```bash
# Solution: Use specific GPUs
CUDA_VISIBLE_DEVICES=2,3 python train_script.py
```

#### 2. Protobuf Compatibility Error
```bash
# Error: AttributeError in simplecv
# Solution: Downgrade protobuf
pip install "protobuf<3.21" --force-reinstall
```

#### 3. Ever Framework Import Error
```bash
# Error: SyntaxError with union operators
# Solution: Ensure Python ≥ 3.8
conda install python=3.9 -y
```

#### 4. SimpleCV Import Error
```bash
# Solution: Install correct version from GitHub
pip uninstall simplecv -y
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

## Performance Optimizations

### Multi-GPU Training
```bash
# Use multiple GPUs for training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py
```

### Memory Optimization
```bash
# Reduce batch size if needed
--batch_size_train 1
--patch_size 512
```

### Data Loading
```bash
# Optimize number of workers
--num_workers 8  # Typically 2x number of GPUs
```

## Environment Recreation

If you need to recreate this environment in the future:

1. **Save current state** (already done):
   ```bash
   conda env export > farsegpp_environment.yml
   pip list --format=freeze > farsegpp_pip_packages.txt
   ```

2. **Recreate from saved state**:
   ```bash
   conda env create -f farsegpp_environment.yml
   ```

3. **Verify functionality**:
   ```bash
   conda activate farsegpp
   python -c "import torch, ever, simplecv; print('✅ Environment working')"
   ```

## Version History

- **August 2025**: Initial working configuration established
  - Python 3.9.23, PyTorch 2.4.0, Ever 0.5.2
  - All FarSeg++ components tested and verified
  - Multi-GPU training validated on 4x RTX 2080 Ti

## Support

For issues with this environment:

1. Check this documentation first
2. Verify all components with the test commands provided
3. Ensure GPU availability with `nvidia-smi`
4. Check CUDA compatibility with `torch.cuda.is_available()`

This environment configuration has been thoroughly tested and validated for production use with both FarSeg and FarSeg++ models.
