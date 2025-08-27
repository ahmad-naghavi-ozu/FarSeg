#!/usr/bin/env python3
"""
Simplified training script for FarSeg without Apex dependency.
Uses native PyTorch AMP for mixed precision training.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import importlib.util
import argparse
from tqdm import tqdm
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def build_model(config):
    """Build the FarSeg model from config."""
    from module.farseg import FarSeg
    return FarSeg(config['model']['params'])

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized fg_cls_label tensors"""
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0])  # image tensor
        targets.append(item[1])  # target dict
    
    # Stack images normally
    batched_images = torch.stack(images, 0)
    
    # Handle target dictionaries with variable-sized tensors
    batched_targets = {}
    
    # For 'cls' - these should all be the same size (segmentation masks)
    cls_list = [target['cls'] for target in targets]
    batched_targets['cls'] = torch.stack(cls_list, 0)
    
    # For 'fg_cls_label' - these can have different sizes, so we need to pad or handle specially
    if 'fg_cls_label' in targets[0]:
        fg_cls_labels = [target['fg_cls_label'] for target in targets]
        
        # Find the maximum length
        max_len = max(len(label) for label in fg_cls_labels) if fg_cls_labels else 0
        
        if max_len > 0:
            # Pad shorter tensors with -1 (ignore index)
            padded_labels = []
            for label in fg_cls_labels:
                if len(label) == 0:
                    # For empty labels, create a tensor with ignore index
                    padded_label = torch.full((max_len,), -1, dtype=label.dtype)
                elif len(label) < max_len:
                    # Pad with -1
                    padding = torch.full((max_len - len(label),), -1, dtype=label.dtype)
                    padded_label = torch.cat([label, padding])
                else:
                    padded_label = label
                padded_labels.append(padded_label)
            
            batched_targets['fg_cls_label'] = torch.stack(padded_labels, 0)
        else:
            # All labels are empty, create a batch of empty tensors
            batch_size = len(targets)
            batched_targets['fg_cls_label'] = torch.full((batch_size, 1), -1, dtype=torch.int64)
    
    return [batched_images, batched_targets]


def build_dataset(config, split='train'):
    """Build dataset from config."""
    from data.generic_dataset import GenericSegmentationDataset
    
    dataset_config = config['data'][split]['params']
    return GenericSegmentationDataset(
        image_dir=dataset_config['image_dir'],
        mask_dir=dataset_config['mask_dir'],
        patch_config=dataset_config['patch_config'],
        transforms=dataset_config['transforms'],
        image_extension='.tif',
        mask_extension='.tif'
    )

def build_optimizer(model, config):
    """Build optimizer from config."""
    optimizer_config = config['optimizer']
    if optimizer_config['type'] == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config['learning_rate']['params']['base_lr'],
            momentum=optimizer_config['params']['momentum'],
            weight_decay=optimizer_config['params']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")

def polynomial_lr_scheduler(optimizer, current_iter, max_iters, base_lr, power=0.9):
    """Polynomial learning rate scheduler."""
    lr = base_lr * (1 - current_iter / max_iters) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def validate_config(config_path):
    """Validate configuration file and data paths."""
    print("🔍 Validating configuration...")
    
    try:
        # Load config
        config = load_config(config_path)
        print(f"✅ Configuration loaded successfully from {config_path}")
        
        # Validate required keys
        required_keys = ['model', 'data', 'optimizer', 'learning_rate', 'train']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required configuration key: {key}")
                return False
        
        print(f"✅ All required configuration keys present")
        
        # Print key configuration info
        model_type = config['model']['type']
        num_classes = config['model']['params']['num_classes']
        batch_size = config['data']['train']['params']['batch_size']
        max_iters = config['train']['num_iters']
        base_lr = config['learning_rate']['params']['base_lr']
        
        print(f"📋 Configuration Summary:")
        print(f"   Model type: {model_type}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max iterations: {max_iters}")
        print(f"   Base learning rate: {base_lr}")
        
        # Validate data paths
        print(f"\n🔍 Validating data paths...")
        train_config = config['data']['train']['params']
        test_config = config['data']['test']['params']
        
        paths_to_check = [
            ('Training images', train_config['image_dir']),
            ('Training masks', train_config['mask_dir']),
            ('Test images', test_config['image_dir']),
            ('Test masks', test_config['mask_dir'])
        ]
        
        all_paths_valid = True
        for name, path in paths_to_check:
            if os.path.exists(path):
                # Count files
                import glob
                file_count = len(glob.glob(os.path.join(path, '*')))
                print(f"✅ {name}: {path} ({file_count} files)")
            else:
                print(f"❌ {name}: {path} (not found)")
                all_paths_valid = False
        
        if all_paths_valid:
            print(f"\n✅ Configuration validation passed!")
            print(f"💡 Ready to start training with: python train_simple.py --config {config_path} --model_dir <model_dir>")
            return True
        else:
            print(f"\n❌ Configuration validation failed - some data paths are missing")
            return False
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def train_model(config_path, model_dir, gpu_ids="0"):
    """Main training function."""
    
    # Clear CUDA cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Setup GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    # Force PyTorch to reinitialize CUDA context with new visible devices
    if torch.cuda.is_available():
        # This ensures PyTorch sees the correct GPU as device 0
        device = torch.device('cuda:0')
        print(f"Using GPU {gpu_ids} (mapped to cuda:0)")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Build model with proper error handling
    print("Building model...")
    model = build_model(config)
    print(f"Model created: {type(model).__name__}")
    
    # Move model to GPU with memory management
    try:
        print(f"Moving model to {device}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before moving model
        model.to(device)
        print(f"✅ Model successfully moved to {device}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ CUDA out of memory: {e}")
            print(f"💡 GPU {gpu_ids} might still have processes running. Try a different GPU or restart.")
            return
        else:
            raise e
    
    # Build datasets
    train_dataset = build_dataset(config, 'train')
    print(f"Training dataset: {len(train_dataset)} samples")
    
    # Get training parameters
    batch_size = config['data']['train']['params']['batch_size']
    
    # Create data loader with better error handling
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing to avoid tensor resize issues
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to avoid size issues
        collate_fn=custom_collate_fn  # Use our custom collate function
    )
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training parameters
    max_iters = config['train']['num_iters']
    log_interval = config['train'].get('log_interval_step', 50)
    save_interval = 5000
    
    # Training loop
    model.train()
    global_step = 0
    epoch = 0
    
    print(f"Starting training for {max_iters} iterations...")
    print(f"Device: {device}")
    print(f"Mixed precision: Enabled")
    print(f"Batch size: {batch_size}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    start_time = time.time()
    
    # Global progress bar for overall training
    pbar = tqdm(total=max_iters, desc="Training", unit="step")
    
    while global_step < max_iters:
        epoch += 1
        
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_iters:
                break
            
            # Handle batch format - batch is [images, targets_dict]
            images = batch[0].to(device)  # tensor
            targets_dict = batch[1]       # dict
            
            # Move all tensors in the targets dict to device
            targets = {}
            for key, value in targets_dict.items():
                if torch.is_tensor(value):
                    targets[key] = value.to(device)
                else:
                    targets[key] = value
            
            # Update learning rate
            lr_config = config['learning_rate']['params']
            current_lr = polynomial_lr_scheduler(
                optimizer, global_step, max_iters, 
                lr_config['base_lr'], lr_config['power']
            )
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast():
                loss_dict = model(images, targets)
                # FarSeg returns a dictionary of losses during training
                if isinstance(loss_dict, dict):
                    total_loss = sum(loss_dict.values())
                else:
                    total_loss = loss_dict
            
            # Backward pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Epoch': epoch,
                'Loss': f'{total_loss.item():.1f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Console logging every log_interval steps
            if global_step % log_interval == 0 and global_step > 0:
                elapsed = time.time() - start_time
                print(f"\nStep {global_step:6d}/{max_iters} | "
                      f"Epoch {epoch:3d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Save checkpoint
            if global_step % save_interval == 0 and global_step > 0:
                checkpoint_path = os.path.join(model_dir, f"model-{global_step}.pth")
                
                # Create a clean state dict for saving (avoid pickle issues)
                model_state = {}
                for key, value in model.state_dict().items():
                    if torch.is_tensor(value):
                        model_state[key] = value.cpu()
                    else:
                        model_state[key] = value
                
                # Save only picklable state information
                torch.save({
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_type': 'FarSeg',
                    'num_classes': getattr(config, 'model', {}).get('num_classes', 2)
                }, checkpoint_path)
                print(f"\n✅ Checkpoint saved: {checkpoint_path}")
            
            global_step += 1
        
        print(f"\nEpoch {epoch} completed")
    
    # Close progress bar
    pbar.close()
    
    # Save final model (avoiding pickle issues with config)
    final_path = os.path.join(model_dir, f"model-{global_step}.pth")
    
    # Create clean state dict for final save
    model_state = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            model_state[key] = value.cpu()
        else:
            model_state[key] = value
    
    # Save only picklable state information
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'epoch': epoch,
        'model_type': 'FarSeg',
        'num_classes': getattr(config, 'model', {}).get('num_classes', 2),
        'training_completed': True
    }, final_path)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final model saved: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='Train FarSeg model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--gpu_ids', default="0", help='GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--validate_config', action='store_true', 
                       help='Only validate configuration and data paths, do not train')
    
    args = parser.parse_args()
    
    # If validation only, run validation and exit
    if args.validate_config:
        success = validate_config(args.config)
        sys.exit(0 if success else 1)
    
    # Otherwise, run training
    train_model(args.config, args.model_dir, args.gpu_ids)

if __name__ == '__main__':
    main()
