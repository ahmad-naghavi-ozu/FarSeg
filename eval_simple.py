#!/usr/bin/env python3
"""
Generic evaluation script for FarSeg models.
Works with any dataset configuration and provides comprehensive metrics.
"""

import os
import sys
import warnings

# Force unbuffered output for real-time progress bars
os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*non-writable.*')
warnings.filterwarnings('ignore', message='.*deprecated.*')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import importlib.util
import argparse
import numpy as np
from tqdm import tqdm
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our BuildFormer-style evaluator
from evaluator import Evaluator

def load_config(config_path):
    """Load configuration from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def build_model(config):
    """Build the FarSeg or FarSegPP model from config."""
    model_type = config['model']['type']
    model_params = config['model']['params']
    
    if model_type == 'FarSegPP':
        from module.farsegpp import FarSegPP
        return FarSegPP(model_params)
    else:
        from module.farseg import FarSeg
        return FarSeg(model_params)

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized tensors"""
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

def build_dataset(config, split='test'):
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

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if file is not empty
    if os.path.getsize(checkpoint_path) < 1000:  # Less than 1KB indicates corruption
        raise RuntimeError(f"Checkpoint file appears to be corrupted (too small): {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully")
        if 'global_step' in checkpoint:
            print(f"   Trained for {checkpoint['global_step']} iterations")
        if 'epoch' in checkpoint:
            print(f"   Completed {checkpoint['epoch']} epochs")
    else:
        # Fallback for older checkpoints
        model.load_state_dict(checkpoint)
        print(f"✅ Model loaded successfully (legacy format)")
    
    return model

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in model directory."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Look for model files
    checkpoint_files = []
    # Check for validation-based training checkpoints first (best_model.pth, latest_model.pth)
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    latest_model_path = os.path.join(model_dir, 'latest_model.pth')
    
    if os.path.exists(best_model_path) and os.path.getsize(best_model_path) > 1000:
        print(f"✅ Found validation-based training checkpoint: best_model.pth")
        return best_model_path, 0  # Return 0 for iteration number (epoch-based training)
    elif os.path.exists(latest_model_path) and os.path.getsize(latest_model_path) > 1000:
        print(f"✅ Found validation-based training checkpoint: latest_model.pth")
        return latest_model_path, 0
    
    # Fall back to iteration-based checkpoints (model-{iter}.pth)
    for file in os.listdir(model_dir):
        if file.startswith('model-') and file.endswith('.pth'):
            file_path = os.path.join(model_dir, file)
            # Skip corrupted or very small files
            if os.path.getsize(file_path) < 1000:  # Less than 1KB
                print(f"⚠️  Skipping corrupted checkpoint: {file} ({os.path.getsize(file_path)} bytes)")
                continue
            
            # Extract iteration number
            try:
                iter_num = int(file.replace('model-', '').replace('.pth', ''))
                checkpoint_files.append((iter_num, file))
            except ValueError:
                continue
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No valid model checkpoints found in: {model_dir}")
    
    # Sort by iteration number and get the latest
    checkpoint_files.sort(key=lambda x: x[0])
    latest_iter, latest_file = checkpoint_files[-1]
    
    return os.path.join(model_dir, latest_file), latest_iter

def evaluate_with_buildformer_style(all_predictions, all_ground_truth, num_classes):
    """
    Evaluate using BuildFormer-style global confusion matrix approach.
    This is the recommended standard approach for segmentation evaluation.
    Now properly handles reconstructed full images for sample-based evaluation.
    """
    print("🔬 Computing metrics using BuildFormer-style global accumulation on reconstructed images...")
    
    # Initialize evaluator
    evaluator = Evaluator(num_class=num_classes)
    
    # Handle both tensor stacks (new sample-based) and lists (legacy)
    if torch.is_tensor(all_predictions):
        # New sample-based approach: reconstructed full images as tensor stack
        print(f"   📊 Processing {len(all_predictions)} reconstructed full images")
        
        for i in tqdm(range(len(all_predictions)), desc="Accumulating confusion matrix", unit="image"):
            pred = all_predictions[i].numpy() if torch.is_tensor(all_predictions[i]) else all_predictions[i]
            gt = all_ground_truth[i].numpy() if torch.is_tensor(all_ground_truth[i]) else all_ground_truth[i]
            
            # Exclude ignore_index pixels (255)
            valid_mask = (gt != 255) & (gt >= 0) & (gt < num_classes)
            if valid_mask.sum() > 0:  # Only process if there are valid pixels
                pred_valid = pred[valid_mask]
                gt_valid = gt[valid_mask]
                evaluator.add_batch(gt_valid, pred_valid)
    else:
        # Legacy patch-based approach (for backward compatibility)
        print(f"   📊 Processing {len(all_predictions)} patches (legacy mode)")
        
        for i in tqdm(range(len(all_predictions)), desc="Accumulating confusion matrix", unit="patch"):
            pred = all_predictions[i].numpy() if torch.is_tensor(all_predictions[i]) else all_predictions[i]
            gt = all_ground_truth[i].numpy() if torch.is_tensor(all_ground_truth[i]) else all_ground_truth[i]
            
            # Exclude ignore_index pixels (255)
            valid_mask = (gt != 255) & (gt >= 0) & (gt < num_classes)
            if valid_mask.sum() > 0:  # Only process if there are valid pixels
                pred_valid = pred[valid_mask]
                gt_valid = gt[valid_mask]
                evaluator.add_batch(gt_valid, pred_valid)
    
    # Get all metrics
    metrics = evaluator.summary()
    
    return {
        'iou_per_class': metrics['iou_per_class'],
        'mean_iou': metrics['miou'],
        'precision_per_class': metrics['precision_per_class'],
        'recall_per_class': metrics['recall_per_class'],
        'f1_per_class': metrics['f1_per_class'],
        'overall_accuracy': metrics['overall_accuracy'],
        'frequency_weighted_iou': metrics['frequency_weighted_iou'],
        'confusion_matrix': metrics['confusion_matrix']
    }

def calculate_iou(pred_mask, true_mask, num_classes, ignore_index=255):
    """Calculate IoU for each class."""
    ious = []
    
    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        # Exclude ignore_index pixels
        valid_pixels = (true_mask != ignore_index)
        pred_class = pred_class & valid_pixels
        true_class = true_class & valid_pixels
        
        intersection = (pred_class & true_class).sum().item()
        union = (pred_class | true_class).sum().item()
        
        if union == 0:
            iou = float('nan')  # No pixels for this class
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return ious

def save_prediction_samples(images, predictions, ground_truth, output_dir, num_samples=5):
    """Save sample predictions for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image (convert from tensor)
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Normalize to 0-1 range for display
        img = (img - img.min()) / (img.max() - img.min())
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Ground truth
        gt = ground_truth[i].cpu().numpy()
        axes[1].imshow(gt, cmap='viridis')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        pred = predictions[i].cpu().numpy()
        axes[2].imshow(pred, cmap='viridis')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Saved {num_samples} prediction samples to: {output_dir}")

def save_prediction_masks(predictions, ground_truth, output_dir, dataset_name, test_dataset, processed_files=None):
    """Save prediction masks with same filenames as ground truth by reconstructing from patches."""
    dataset_predictions_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_predictions_dir, exist_ok=True)
    
    print(f"💾 Saving prediction masks to: {dataset_predictions_dir}")
    
    # Get unique image files from dataset _data_list
    try:
        if hasattr(test_dataset, '_data_list') and test_dataset._data_list:
            # Extract unique mask paths and filenames
            unique_files = {}
            for item in test_dataset._data_list:
                image_path, mask_path, window = item
                if mask_path:
                    filename = os.path.basename(mask_path)
                    if filename not in unique_files:
                        unique_files[filename] = {
                            'mask_path': mask_path,
                            'patches': []
                        }
                    unique_files[filename]['patches'].append({
                        'window': window,
                        'index': len(unique_files[filename]['patches'])
                    })
            
            print(f"Found {len(unique_files)} unique images with {len(predictions)} total patches")
            
            # Reconstruct full images from patches
            patch_idx = 0
            for filename, file_info in unique_files.items():
                mask_path = file_info['mask_path']
                patches_info = file_info['patches']
                
                # Load original mask to get dimensions
                original_mask = Image.open(mask_path)
                width, height = original_mask.size
                
                # Create empty prediction image
                full_prediction = np.zeros((height, width), dtype=np.uint8)
                patch_count = np.zeros((height, width), dtype=np.uint8)
                
                # Aggregate patches
                for patch_info in patches_info:
                    if patch_idx < len(predictions):
                        window = patch_info['window']
                        pred_patch = predictions[patch_idx].cpu().numpy().astype(np.uint8)
                        
                        # Extract window coordinates
                        y1, x1, y2, x2 = window
                        
                        # Add patch to full image
                        full_prediction[y1:y2, x1:x2] += pred_patch
                        patch_count[y1:y2, x1:x2] += 1
                        
                        patch_idx += 1
                
                # Average overlapping regions
                mask = patch_count > 0
                full_prediction[mask] = full_prediction[mask] // patch_count[mask]
                
                # Save reconstructed prediction
                pred_path = os.path.join(dataset_predictions_dir, filename)
                Image.fromarray(full_prediction).save(pred_path)
            
            print(f"✅ Saved {len(unique_files)} prediction masks with original filenames")
            
        else:
            # Fallback: save patches with numbered names
            print("⚠️  Warning: Could not extract filenames from dataset, saving patches individually")
            for i, pred_mask in enumerate(predictions):
                pred_array = pred_mask.cpu().numpy().astype(np.uint8)
                pred_path = os.path.join(dataset_predictions_dir, f"prediction_patch_{i+1:04d}.png")
                Image.fromarray(pred_array).save(pred_path)
            
            print(f"✅ Saved {len(predictions)} prediction patches")
            
    except Exception as e:
        print(f"❌ Error during prediction saving: {e}")
        # Emergency fallback
        for i, pred_mask in enumerate(predictions):
            pred_array = pred_mask.cpu().numpy().astype(np.uint8)
            pred_path = os.path.join(dataset_predictions_dir, f"prediction_{i+1:04d}.png")
            Image.fromarray(pred_array).save(pred_path)
        
        print(f"✅ Saved {len(predictions)} prediction masks (fallback mode)")
    
    return dataset_predictions_dir

def reconstruct_image_from_patches(patch_predictions, patch_info_list, original_height, original_width):
    """Reconstruct full image prediction from overlapping patches."""
    full_prediction = np.zeros((original_height, original_width), dtype=np.float32)
    patch_count = np.zeros((original_height, original_width), dtype=np.float32)
    
    for i, patch_info in enumerate(patch_info_list):
        if i < len(patch_predictions):
            window = patch_info['window']
            pred_patch = patch_predictions[i].cpu().numpy().astype(np.float32)
            
            # Extract window coordinates
            y1, x1, y2, x2 = window
            
            # Add patch to full image
            full_prediction[y1:y2, x1:x2] += pred_patch
            patch_count[y1:y2, x1:x2] += 1
    
    # Average overlapping regions
    mask = patch_count > 0
    full_prediction[mask] = full_prediction[mask] / patch_count[mask]
    
    return full_prediction.astype(np.uint8)

def evaluate_sample_based(test_dataset, model, device, num_classes):
    """
    Sample-based evaluation: reconstruct full images from patches and compute metrics.
    This is the correct way to evaluate segmentation models trained on patches.
    """
    print("🔍 Starting sample-based evaluation (reconstructing full images from patches)...")
    
    # Group patches by original image
    unique_files = {}
    for idx, item in enumerate(test_dataset._data_list):
        image_path, mask_path, window = item
        if mask_path:
            filename = os.path.basename(mask_path)
            if filename not in unique_files:
                unique_files[filename] = {
                    'mask_path': mask_path,
                    'image_path': image_path,
                    'patches': []
                }
            unique_files[filename]['patches'].append({
                'window': window,
                'dataset_index': idx
            })
    
    print(f"Found {len(unique_files)} unique images with {len(test_dataset)} total patches")
    
    all_full_predictions = []
    all_full_ground_truth = []
    all_full_images = []  # Store reconstructed images
    
    # Process each unique image
    for filename, file_info in tqdm(unique_files.items(), desc="Processing images", unit="image"):
        mask_path = file_info['mask_path']
        image_path = file_info['image_path']
        patches_info = file_info['patches']
        
        # Load original mask to get dimensions
        original_mask = Image.open(mask_path)
        width, height = original_mask.size
        original_gt = np.array(original_mask)
        
        # Load original image for visualization
        original_image = Image.open(image_path)
        original_image_array = np.array(original_image)
        
        # Get predictions for all patches of this image
        patch_predictions = []
        
        with torch.no_grad():
            for patch_info in patches_info:
                dataset_idx = patch_info['dataset_index']
                
                # Get patch from dataset
                image_patch, target_patch = test_dataset[dataset_idx]
                
                # Add batch dimension and move to device
                image_batch = image_patch.unsqueeze(0).to(device)
                
                # Forward pass
                try:
                    outputs = model(image_batch)
                    if isinstance(outputs, dict):
                        logits = outputs.get('seg', outputs.get('cls', outputs))
                    else:
                        logits = outputs
                except:
                    # Fallback with targets if needed
                    target_batch = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else v 
                                  for k, v in target_patch.items()}
                    outputs = model(image_batch, target_batch)
                    if isinstance(outputs, dict):
                        if 'pred' in outputs:
                            logits = outputs['pred']
                        elif 'seg' in outputs:
                            logits = outputs['seg']
                        elif 'cls' in outputs:
                            logits = outputs['cls']
                        else:
                            outputs = model(image_batch)
                            logits = outputs
                    else:
                        logits = outputs
                
                prediction = torch.argmax(logits, dim=1).squeeze(0)
                patch_predictions.append(prediction)
        
        # Reconstruct full image from patches
        full_prediction = reconstruct_image_from_patches(
            patch_predictions, patches_info, height, width
        )
        
        all_full_predictions.append(full_prediction)
        all_full_ground_truth.append(original_gt)
        all_full_images.append(original_image_array)
    
    print(f"✅ Reconstructed {len(all_full_predictions)} full images from patches")
    
    # Convert to tensors for metric calculation
    all_predictions_tensor = torch.stack([torch.from_numpy(pred) for pred in all_full_predictions])
    all_ground_truth_tensor = torch.stack([torch.from_numpy(gt) for gt in all_full_ground_truth])
    
    # Compute metrics on reconstructed full images
    buildformer_metrics = evaluate_with_buildformer_style(
        all_predictions_tensor, all_ground_truth_tensor, num_classes
    )
    
    return buildformer_metrics, all_full_predictions, all_full_ground_truth, all_full_images, len(unique_files)

def evaluate_model(config_path, model_dir, output_dir, gpu_ids="0", checkpoint_path=None, force_predictions=False):
    """Main evaluation function with sample-based reconstruction."""
    
    # Setup GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU {gpu_ids} (mapped to cuda:0)")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    
    # Determine model type from config
    model_type = config['model']['type']
    if model_type == 'FarSegPP':
        model_name = 'FarSeg++'
    else:
        model_name = 'FarSeg'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.to(device)
    model.eval()
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path, trained_iters = find_latest_checkpoint(model_dir)
        print(f"Found latest checkpoint: {trained_iters} iterations")
    else:
        trained_iters = "unknown"
    
    model = load_model_checkpoint(model, checkpoint_path, device)
    
    # Build test dataset
    print("📚 Building test dataset...")
    test_dataset = build_dataset(config, 'test')
    print(f"✅ Test dataset: {len(test_dataset)} patches from individual images")
    
    # Get evaluation parameters
    num_classes = config['model']['params']['num_classes']
    
    # ========================================
    # SAMPLE-BASED EVALUATION (CORRECT APPROACH)
    # ========================================
    print(f"🔬 Starting SAMPLE-BASED evaluation...")
    print(f"This will reconstruct full images from overlapping patches.")
    print(f"Device: {device}")
    print(f"Number of classes: {num_classes}")
    
    start_time = time.time()
    
    # Perform sample-based evaluation
    buildformer_metrics, all_full_predictions, all_full_ground_truth, all_full_images, num_images = evaluate_sample_based(
        test_dataset, model, device, num_classes
    )
    
    evaluation_time = time.time() - start_time
    
    # Convert reconstructed predictions to tensor format for saving
    all_predictions_tensor = torch.stack([torch.from_numpy(pred) for pred in all_full_predictions])
    all_ground_truth_tensor = torch.stack([torch.from_numpy(gt) for gt in all_full_ground_truth])
    
    # For visualization, use the actual reconstructed images
    sample_images = []
    sample_predictions = []
    sample_ground_truth = []
    
    for i in range(min(5, len(all_full_predictions))):
        # Convert numpy image to tensor (H, W, C) -> (C, H, W)
        image_array = all_full_images[i]
        if len(image_array.shape) == 2:  # Grayscale
            image_tensor = torch.from_numpy(image_array).unsqueeze(0).float()
        else:  # RGB
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        sample_images.append(image_tensor)
        sample_predictions.append(all_predictions_tensor[i])
        sample_ground_truth.append(all_ground_truth_tensor[i])
    
    if sample_images:
        sample_images_tensor = torch.stack(sample_images)
        sample_predictions_tensor = torch.stack(sample_predictions)  
        sample_ground_truth_tensor = torch.stack(sample_ground_truth)
    else:
        # Create empty tensors as fallback
        sample_images_tensor = torch.zeros(1, 3, 512, 512)
        sample_predictions_tensor = torch.zeros(1, 512, 512)
        sample_ground_truth_tensor = torch.zeros(1, 512, 512)
    
    # Use sample-based metrics
    mean_iou_per_class = buildformer_metrics['iou_per_class']
    mean_iou = buildformer_metrics['mean_iou']
    conf_matrix = buildformer_metrics['confusion_matrix']

    # Get valid predictions and ground truth for sklearn metrics (flattened)
    all_valid_preds = []
    all_valid_gt = []
    
    for pred, gt in zip(all_full_predictions, all_full_ground_truth):
        valid_mask = (gt != 255)  # Ignore index
        if valid_mask.sum() > 0:
            all_valid_preds.extend(pred[valid_mask].tolist())
            all_valid_gt.extend(gt[valid_mask].tolist())
    
    # Convert to numpy for sklearn
    valid_predictions = np.array(all_valid_preds)
    valid_ground_truth = np.array(all_valid_gt)

    # Classification report
    print("📋 Generating classification report...")
    class_report = classification_report(
        valid_ground_truth,
        valid_predictions,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )

    # Calculate per-class metrics using sklearn (for compatibility)
    print("📈 Computing precision, recall, and F1 scores...")
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_ground_truth,
        valid_predictions,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS")
    print(f"=" * 80)
    print(f"Dataset: {os.path.basename(config_path).replace('farseg_', '').replace('.py', '')}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Test samples: {num_images}")  # Now counts reconstructed images
    print(f"Evaluation time: {evaluation_time:.1f}s")
    print(f"")
    print(f"Overall Metrics:")
    print(f"  Overall Accuracy: {buildformer_metrics['overall_accuracy']:.4f}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Frequency Weighted IoU: {buildformer_metrics['frequency_weighted_iou']:.4f}")
    print(f"")
    print(f"Per-Class Metrics:")
    for class_id in range(num_classes):
        iou = buildformer_metrics['iou_per_class'][class_id]
        
        print(f"  Class {class_id}:")
        print(f"    IoU: {iou:.4f}")
        print(f"    Precision: {buildformer_metrics['precision_per_class'][class_id]:.4f}")
        print(f"    Recall: {buildformer_metrics['recall_per_class'][class_id]:.4f}")
        print(f"    F1-Score: {buildformer_metrics['f1_per_class'][class_id]:.4f}")
        print(f"    Support: {support[class_id]}")
    
    # Save detailed results
    results = {
        'dataset': os.path.basename(config_path).replace('farseg_', '').replace('.py', ''),
        'model': model_name,
        'checkpoint': os.path.basename(checkpoint_path),
        'trained_iterations': trained_iters,
        'test_samples': num_images,  # Now counts reconstructed images
        'evaluation_time': evaluation_time,
        'metrics': {
            'overall_accuracy': buildformer_metrics['overall_accuracy'],
            'mean_iou': mean_iou,
            'frequency_weighted_iou': buildformer_metrics['frequency_weighted_iou'],
            'per_class_iou': buildformer_metrics['iou_per_class'].tolist(),
            'per_class_precision': buildformer_metrics['precision_per_class'].tolist(),
            'per_class_recall': buildformer_metrics['recall_per_class'].tolist(),
            'per_class_f1': buildformer_metrics['f1_per_class'].tolist(),
            'per_class_support': support.tolist()
        },
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'eval_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save results to text file
    results_txt = os.path.join(output_dir, 'eval_results.txt')
    with open(results_txt, 'w') as f:
        f.write(f"{model_name} Evaluation Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"Checkpoint: {results['checkpoint']}\n")
        f.write(f"Test samples: {results['test_samples']}\n")
        f.write(f"Evaluation time: {results['evaluation_time']:.1f}s\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"  Overall Accuracy: {results['metrics']['overall_accuracy']:.4f}\n")
        f.write(f"  Mean IoU: {results['metrics']['mean_iou']:.4f}\n")
        f.write(f"  Frequency Weighted IoU: {results['metrics']['frequency_weighted_iou']:.4f}\n")
        f.write(f"\nPer-Class Metrics:\n")
        for class_id in range(num_classes):
            f.write(f"  Class {class_id}:\n")
            f.write(f"    IoU: {results['metrics']['per_class_iou'][class_id]:.4f}\n")
            f.write(f"    Precision: {results['metrics']['per_class_precision'][class_id]:.4f}\n")
            f.write(f"    Recall: {results['metrics']['per_class_recall'][class_id]:.4f}\n")
            f.write(f"    F1-Score: {results['metrics']['per_class_f1'][class_id]:.4f}\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    # Convert to int to avoid float formatting issues
    conf_matrix_int = conf_matrix.astype(int)
    sns.heatmap(conf_matrix_int, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    confusion_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save sample predictions for evaluation visualization
    samples_dir = os.path.join(output_dir, 'prediction_samples')
    save_prediction_samples(sample_images_tensor, sample_predictions_tensor, sample_ground_truth_tensor, samples_dir)
    
    # Save prediction masks to output directory (model-specific structure)
    # Create predictions subdirectory within the model-specific output directory
    predictions_dir = os.path.join(output_dir, 'predictions')
    
    # Check if we should save predictions
    existing_files = []
    if os.path.exists(predictions_dir):
        existing_files = [f for f in os.listdir(predictions_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    should_save_predictions = force_predictions or len(existing_files) == 0
    
    if should_save_predictions:
        final_predictions_dir = save_prediction_masks(all_predictions_tensor, all_ground_truth_tensor, output_dir, 'predictions', test_dataset)
    else:
        final_predictions_dir = predictions_dir
        print(f"Skipping prediction saving - predictions already exist (use --force_predictions to regenerate)")
    
    print(f"\n💾 Results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {results_txt}")
    print(f"  Confusion matrix: {confusion_matrix_file}")
    print(f"  Sample predictions: {samples_dir}")
    print(f"  Prediction masks: {final_predictions_dir}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate FarSeg model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model_dir', required=True, help='Model directory containing checkpoints')
    parser.add_argument('--output_dir', help='Output directory for evaluation results (default: model_dir/evaluation)')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to evaluate (default: latest)')
    parser.add_argument('--gpu_ids', default="0", help='GPU IDs (e.g., "0,1,2,3")')
    parser.add_argument('--force_predictions', action='store_true', help='Force regeneration of predictions even if they exist')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_dir, 'evaluation')
    
    # Run evaluation
    evaluate_model(
        config_path=args.config,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        gpu_ids=args.gpu_ids,
        checkpoint_path=args.checkpoint,
        force_predictions=args.force_predictions
    )

if __name__ == '__main__':
    main()
