#!/usr/bin/env python3
"""
Generic training and evaluation script for FarSeg with any dataset.

Usage:
    # Generate config first
    python generate_config.py --dataset_name my_dataset --num_classes 5 --data_root /path/to/data
    
    # Train the model
    python train_generic.py --config configs/my_dataset/farseg_my_dataset.py --model_dir ./models/my_dataset
    
    # Evaluate the model
    python train_generic.py --config configs/my_dataset/farseg_my_dataset.py --model_dir ./models/my_dataset --eval_only
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
import json
import numpy as np
from simplecv.util import logger, registry
from simplecv.util.checkpoint import CheckPointer

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import generic_dataset
from module import farseg


def setup_environment():
    """Setup the training environment."""
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def train_model(config_path, model_dir, local_rank=0, opt_level='O1', cpu_mode=False, opts=None):
    """Train the FarSeg model with given configuration."""
    print(f"Starting training with config: {config_path}")
    print(f"Model will be saved to: {model_dir}")
    
    setup_environment()
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Start training
    train.run(
        local_rank=local_rank,
        config_path=config_path,
        model_dir=model_dir,
        opt_level=opt_level,
        cpu_mode=cpu_mode,
        after_construct_launcher_callbacks=[],
        opts=opts or []
    )


def evaluate_model(config_path, model_dir, checkpoint_path=None):
    """Evaluate the trained model."""
    print(f"Starting evaluation with config: {config_path}")
    print(f"Model directory: {model_dir}")
    
    setup_environment()
    
    # Find the best checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            print(f"No checkpoint files found in {model_dir}")
            return
        
        # Use the latest checkpoint
        checkpoint_files.sort()
        checkpoint_path = os.path.join(model_dir, checkpoint_files[-1])
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Import evaluation script (you might need to adapt this based on your evaluation needs)
    from simplecv.api import eval_segm
    
    # Run evaluation
    eval_segm.run(
        config_path=config_path,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path
    )


def validate_config(config_path):
    """Validate that the configuration file exists and is valid."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return False
    
    try:
        # Try to import the config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if not hasattr(config_module, 'config'):
            print(f"Error: Configuration file must contain a 'config' variable")
            return False
        
        config = config_module.config
        
        # Basic validation
        required_keys = ['model', 'data', 'optimizer', 'learning_rate', 'train']
        for key in required_keys:
            if key not in config:
                print(f"Error: Missing required configuration key: {key}")
                return False
        
        print(f"✅ Configuration validation passed")
        
        # Print some key info
        model_type = config['model']['type']
        num_classes = config['model']['params']['num_classes']
        train_dir = config['data']['train']['params']['image_dir']
        
        print(f"   Model type: {model_type}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Training data: {train_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False


def check_data_paths(config_path):
    """Check if the data paths in the configuration exist."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
        
        train_image_dir = config['data']['train']['params']['image_dir']
        train_mask_dir = config['data']['train']['params']['mask_dir']
        test_image_dir = config['data']['test']['params']['image_dir']
        test_mask_dir = config['data']['test']['params']['mask_dir']
        
        paths_to_check = [
            ('Training images', train_image_dir),
            ('Training masks', train_mask_dir),
            ('Test images', test_image_dir),
            ('Test masks', test_mask_dir)
        ]
        
        all_exist = True
        for name, path in paths_to_check:
            if os.path.exists(path):
                # Count files in directory - support multiple image formats
                import glob
                image_extensions = ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.bmp', '*.gif']
                total_files = 0
                for ext in image_extensions:
                    files = glob.glob(os.path.join(path, ext))
                    files.extend(glob.glob(os.path.join(path, ext.upper())))  # Also check uppercase
                    total_files += len(files)
                print(f"✅ {name}: {path} ({total_files} files)")
            else:
                print(f"❌ {name}: {path} (not found)")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"Error checking data paths: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train/Evaluate FarSeg on generic datasets')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory to save model checkpoints')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--opt_level', type=str, default='O1',
                       choices=['O0', 'O1', 'O2', 'O3'],
                       help='Apex optimization level')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU mode (for debugging)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation, do not train')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint file for evaluation')
    parser.add_argument('--validate_config', action='store_true',
                       help='Only validate configuration and check data paths')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                       help='Override configuration options')
    
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_config(args.config):
        return 1
    
    # Check data paths
    if not check_data_paths(args.config):
        print("\\n⚠️  Some data paths are missing. Please check your dataset setup.")
        if not args.validate_config:
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return 1
    
    # If only validating, exit here
    if args.validate_config:
        print("\\n✅ Configuration and data path validation completed.")
        return 0
    
    try:
        if args.eval_only:
            evaluate_model(args.config, args.model_dir, args.checkpoint)
        else:
            train_model(
                args.config, 
                args.model_dir, 
                args.local_rank, 
                args.opt_level, 
                args.cpu, 
                args.opts
            )
    except Exception as e:
        print(f"Error during {'evaluation' if args.eval_only else 'training'}: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
