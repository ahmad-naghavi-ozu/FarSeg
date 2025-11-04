#!/usr/bin/env python3
"""
Enhanced Training Script for FarSeg/FarSeg++ with Validation Strategy

Implements:
- Validation during training
- Early stopping with patience
- Adaptive learning rate scheduling  
- Best model checkpoint saving
- Comprehensive metrics tracking
"""

import os
import sys
import argparse
import time
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*non-writable.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import importlib.util
from tqdm import tqdm
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from module.early_stopping import EarlyStopping, MetricTracker
from module.metrics import SegmentationMetrics, validate_model
from module.lr_scheduler import build_lr_scheduler, get_current_lr


def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def build_model(config):
    """Build FarSeg or FarSegPP model from config."""
    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'FarSeg':
        from module.farseg import FarSeg
        return FarSeg(model_params)
    elif model_type == 'FarSegPP':
        from module.farsegpp import FarSegPP
        return FarSegPP(model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors."""
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    
    batched_images = torch.stack(images, 0)
    batched_targets = {}
    
    # Handle 'cls' field
    cls_list = [target['cls'] for target in targets]
    batched_targets['cls'] = torch.stack(cls_list, 0)
    
    # Handle 'fg_cls_label' field with padding
    if 'fg_cls_label' in targets[0]:
        fg_cls_labels = [target['fg_cls_label'] for target in targets]
        max_len = max(len(label) for label in fg_cls_labels) if fg_cls_labels else 0
        
        if max_len > 0:
            padded_labels = []
            for label in fg_cls_labels:
                if len(label) == 0:
                    padded_label = torch.full((max_len,), -1, dtype=label.dtype)
                elif len(label) < max_len:
                    padding = torch.full((max_len - len(label),), -1, dtype=label.dtype)
                    padded_label = torch.cat([label, padding])
                else:
                    padded_label = label
                padded_labels.append(padded_label)
            batched_targets['fg_cls_label'] = torch.stack(padded_labels, 0)
        else:
            batch_size = len(targets)
            batched_targets['fg_cls_label'] = torch.full((batch_size, 1), -1, dtype=torch.int64)
    
    return [batched_images, batched_targets]


def build_dataset(config, split='train'):
    """Build dataset from config."""
    from data.generic_dataset import GenericSegmentationDataset, GenericFusionSegmentationDataset
    
    dataset_config = config['data'][split]['params']
    dataloader_type = config['data'][split]['type']
    
    if dataloader_type == 'GenericFusionSegmentationDataLoader':
        image_dirs = dataset_config['image_dir']
        mask_dirs = dataset_config['mask_dir']
        
        if isinstance(image_dirs, str):
            import ast
            image_dirs = ast.literal_eval(image_dirs)
        if isinstance(mask_dirs, str):
            import ast
            mask_dirs = ast.literal_eval(mask_dirs)
            
        return GenericFusionSegmentationDataset(
            image_dirs=image_dirs,
            mask_dirs=mask_dirs,
            patch_config=dataset_config['patch_config'],
            transforms=dataset_config['transforms'],
            image_extension='.tif',
            mask_extension='.tif'
        )
    else:
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
    lr_config = config['learning_rate']
    
    if optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr_config['params']['base_lr'],
            momentum=optimizer_config['params']['momentum'],
            weight_decay=optimizer_config['params']['weight_decay']
        )
    elif optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr_config['params']['base_lr'],
            weight_decay=optimizer_config['params'].get('weight_decay', 0.0)
        )
    elif optimizer_config['type'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr_config['params']['base_lr'],
            weight_decay=optimizer_config['params'].get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    return optimizer


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, global_step, 
                   best_val_miou, metric_tracker, early_stopping, config):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_miou': best_val_miou,
        'metric_history': metric_tracker.get_summary(),
        'early_stopping_counter': early_stopping.counter,
        'model_type': config['model']['type'],
        'num_classes': config['model']['params']['num_classes'],
    }
    
    # Add scheduler state if not ReduceLROnPlateau
    if hasattr(scheduler, 'state_dict') and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint and resume training."""
    print(f"🔄 Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint and hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_val_miou = checkpoint.get('best_val_miou', 0.0)
    
    print(f"✅ Resumed from epoch {epoch}, step {global_step}")
    print(f"   Best validation mIoU: {best_val_miou:.4f}")
    
    return epoch, global_step, best_val_miou


def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, global_step, 
                    log_interval, pbar):
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    
    for batch_idx, batch in enumerate(train_loader):
        # Extract batch data
        images = batch[0].to(device)
        targets_dict = batch[1]
        targets = {k: v.to(device) if torch.is_tensor(v) else v 
                  for k, v in targets_dict.items()}
        
        # Forward pass with mixed precision
        optimizer.zero_grad()
        
        with autocast():
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                total_loss = sum(loss_dict.values())
            else:
                total_loss = loss_dict
        
        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Record loss
        epoch_losses.append(total_loss.item())
        
        # Update progress bar
        pbar.update(1)
        current_lr = get_current_lr(optimizer)
        pbar.set_postfix({
            'Epoch': epoch,
            'Loss': f'{total_loss.item():.4f}',
            'LR': f'{current_lr:.6f}'
        })
        
        # Logging
        if global_step % log_interval == 0 and global_step > 0:
            print(f"\n[Train] Step {global_step} | Epoch {epoch} | "
                  f"Loss: {total_loss.item():.4f} | LR: {current_lr:.6f}")
        
        global_step += 1
    
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    return avg_epoch_loss, global_step


def validate_epoch(model, val_loader, device, num_classes, ignore_index):
    """Run validation and compute metrics."""
    print("\n🔍 Running validation...")
    metrics = validate_model(model, val_loader, device, num_classes, ignore_index)
    
    print(f"✅ Validation completed")
    print(f"   Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"   Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    if 'avg_loss' in metrics:
        print(f"   Avg Loss: {metrics['avg_loss']:.4f}")
    
    return metrics


def train_with_validation(
    config_path,
    model_dir,
    gpu_ids="0",
    validation_interval_epochs=1,
    validation_interval_steps=None,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    lr_scheduler_type=None,
    resume_from=None,
    max_iters_override=None,
    max_epochs_override=None
):
    """
    Main training function with validation strategy.
    
    Args:
        config_path (str): Path to configuration file
        model_dir (str): Directory to save models and checkpoints
        gpu_ids (str): GPU IDs to use
        validation_interval_epochs (int): Validate every N epochs
        validation_interval_steps (int): Validate every N steps (overrides epochs if set)
        early_stopping_patience (int): Number of validations without improvement before stopping
        early_stopping_min_delta (float): Minimum improvement threshold
        lr_scheduler_type (str): Override LR scheduler type ('plateau', 'poly', 'cosine_warmup')
        resume_from (str): Path to checkpoint to resume from
        max_iters_override (int): Override max iterations from config
        max_epochs_override (int): Override max epochs
    """
    
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training with validation strategy")
    print(f"Device: {device}")
    
    # Load config
    config = load_config(config_path)
    print(f"✅ Configuration loaded from {config_path}")
    
    # Override settings if specified
    if max_iters_override:
        config['train']['num_iters'] = max_iters_override
    if lr_scheduler_type:
        config['learning_rate']['type'] = lr_scheduler_type
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Build model
    print("🏗️  Building model...")
    model = build_model(config).to(device)
    print(f"✅ Model: {config['model']['type']}")
    
    # Build datasets
    print("📦 Loading datasets...")
    train_dataset = build_dataset(config, 'train')
    print(f"   Training samples: {len(train_dataset)}")
    
    # Check if validation split exists
    has_validation = 'valid' in config['data'] or 'val' in config['data']
    if has_validation:
        val_split = 'valid' if 'valid' in config['data'] else 'val'
        val_dataset = build_dataset(config, val_split)
        print(f"   Validation samples: {len(val_dataset)}")
    else:
        print("   ⚠️  No validation split found in config. Validation disabled.")
        validation_interval_epochs = None
        validation_interval_steps = None
    
    # Create data loaders
    batch_size = config['data']['train']['params']['batch_size']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['data']['train']['params'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    if has_validation:
        val_batch_size = config['data'][val_split]['params'].get('batch_size', 1)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config['data'][val_split]['params'].get('num_workers', 0),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build LR scheduler
    max_iters = config['train']['num_iters']
    scheduler = build_lr_scheduler(optimizer, config['learning_rate'], max_iters)
    is_plateau_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Initialize tracking
    metric_tracker = MetricTracker()
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode='max',
        verbose=True
    ) if has_validation else None
    
    # Training parameters
    log_interval = config['train'].get('log_interval_step', 50)
    num_classes = config['model']['params']['num_classes']
    ignore_index = config['model']['params'].get('ignore_index', 255)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_miou = 0.0
    
    if resume_from:
        if os.path.isfile(resume_from):
            start_epoch, global_step, best_val_miou = load_checkpoint(
                resume_from, model, optimizer, scheduler, device
            )
        else:
            print(f"⚠️  Checkpoint not found: {resume_from}. Starting from scratch.")
    
    # Calculate training loop parameters
    steps_per_epoch = len(train_loader)
    if max_epochs_override:
        max_epochs = max_epochs_override
    else:
        max_epochs = (max_iters // steps_per_epoch) + 1
    
    print(f"\n📋 Training Configuration:")
    print(f"   Max iterations: {max_iters}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Batch size: {batch_size}")
    print(f"   Initial LR: {get_current_lr(optimizer):.6f}")
    print(f"   LR Scheduler: {config['learning_rate']['type']}")
    if has_validation:
        print(f"   Validation interval: Every {validation_interval_epochs} epoch(s)")
        print(f"   Early stopping patience: {early_stopping_patience}")
    print()
    
    # Training loop
    start_time = time.time()
    pbar = tqdm(initial=global_step, total=max_iters, desc="Training", unit="step")
    
    for epoch in range(start_epoch, max_epochs):
        if global_step >= max_iters:
            print(f"\n✅ Reached maximum iterations ({max_iters})")
            break
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{max_epochs}")
        print(f"{'='*80}")
        
        # Train one epoch
        avg_train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch + 1, global_step, log_interval, pbar
        )
        
        # Update LR scheduler (if not plateau-based)
        if not is_plateau_scheduler:
            scheduler.step()
        
        current_lr = get_current_lr(optimizer)
        print(f"\n[Epoch {epoch + 1}] Average training loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")
        
        # Validation
        should_validate = False
        if has_validation:
            if validation_interval_steps:
                should_validate = (global_step % validation_interval_steps) == 0
            elif validation_interval_epochs:
                should_validate = ((epoch + 1) % validation_interval_epochs) == 0
        
        if should_validate:
            val_metrics = validate_epoch(model, val_loader, device, num_classes, ignore_index)
            val_miou = val_metrics['mean_iou']
            val_loss = val_metrics.get('avg_loss', 0.0)
            
            # Update metric tracker
            metric_tracker.update(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_miou=val_miou,
                val_per_class_iou=val_metrics['per_class_iou'],
                val_pixel_acc=val_metrics['pixel_accuracy'],
                lr=current_lr
            )
            
            # Update plateau scheduler if applicable
            if is_plateau_scheduler:
                scheduler.step(val_miou)
            
            # Save best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_model_path = os.path.join(model_dir, 'best_model.pth')
                save_checkpoint(
                    best_model_path, model, optimizer, scheduler, epoch + 1,
                    global_step, best_val_miou, metric_tracker, early_stopping, config
                )
                print(f"🏆 New best model! Validation mIoU: {best_val_miou:.4f}")
            
            # Check early stopping
            if early_stopping(val_miou, model, epoch + 1):
                print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                print(f"   Best validation mIoU: {best_val_miou:.4f}")
                
                # Save early stop checkpoint
                early_stop_path = os.path.join(model_dir, 'early_stop_model.pth')
                save_checkpoint(
                    early_stop_path, model, optimizer, scheduler, epoch + 1,
                    global_step, best_val_miou, metric_tracker, early_stopping, config
                )
                break
        else:
            # No validation - just track training loss
            metric_tracker.update(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                lr=current_lr
            )
        
        # Save latest checkpoint
        latest_path = os.path.join(model_dir, 'latest_model.pth')
        save_checkpoint(
            latest_path, model, optimizer, scheduler, epoch + 1,
            global_step, best_val_miou, metric_tracker, early_stopping, config
        )
        
        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            periodic_path = os.path.join(model_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(
                periodic_path, model, optimizer, scheduler, epoch + 1,
                global_step, best_val_miou, metric_tracker, early_stopping, config
            )
    
    pbar.close()
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🎉 Training Completed!")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Total epochs: {epoch + 1}")
    print(f"Total steps: {global_step}")
    if has_validation:
        print(f"Best validation mIoU: {best_val_miou:.4f}")
    
    # Save final metrics
    metrics_path = os.path.join(model_dir, 'training_metrics.pth')
    metric_tracker.save(metrics_path)
    
    # Save metrics as JSON for easy viewing
    metrics_json_path = os.path.join(model_dir, 'training_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metric_tracker.get_summary(), f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    print(f"\n📊 Training metrics saved to:")
    print(f"   {metrics_path}")
    print(f"   {metrics_json_path}")
    print(f"\n💾 Model checkpoints saved in: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train FarSeg/FarSeg++ with Validation Strategy')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model_dir', required=True, help='Model directory')
    parser.add_argument('--gpu_ids', default="0", help='GPU IDs')
    parser.add_argument('--validation_interval_epochs', type=int, default=1,
                       help='Validate every N epochs')
    parser.add_argument('--validation_interval_steps', type=int, default=None,
                       help='Validate every N steps (overrides epochs)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                       help='Minimum improvement threshold')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                       choices=['plateau', 'poly', 'poly_warmup', 'cosine_warmup'],
                       help='Override LR scheduler type')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--max_iters', type=int, default=None,
                       help='Override max iterations')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Override max epochs')
    
    args = parser.parse_args()
    
    train_with_validation(
        config_path=args.config,
        model_dir=args.model_dir,
        gpu_ids=args.gpu_ids,
        validation_interval_epochs=args.validation_interval_epochs,
        validation_interval_steps=args.validation_interval_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        lr_scheduler_type=args.lr_scheduler,
        resume_from=args.resume_from,
        max_iters_override=args.max_iters,
        max_epochs_override=args.max_epochs
    )


if __name__ == '__main__':
    main()
