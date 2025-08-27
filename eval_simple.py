#!/usr/bin/env python3
"""
Generic evaluation script for FarSeg models.
Works with any dataset configuration and provides comprehensive metrics.
"""

import os
import sys
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

def evaluate_model(config_path, model_dir, output_dir, gpu_ids="0", checkpoint_path=None):
    """Main evaluation function."""
    
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
    test_dataset = build_dataset(config, 'test')
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Get evaluation parameters
    batch_size = config['data']['test']['params']['batch_size']
    num_classes = config['model']['params']['num_classes']
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # Evaluation metrics
    all_predictions = []
    all_ground_truth = []
    all_images = []
    total_loss = 0.0
    sample_count = 0
    
    print(f"Starting evaluation...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle batch format
            images = batch[0].to(device)
            targets_dict = batch[1]
            
            # Move targets to device
            targets = {}
            for key, value in targets_dict.items():
                if torch.is_tensor(value):
                    targets[key] = value.to(device)
                else:
                    targets[key] = value
            
            # Forward pass - handle inference mode properly
            try:
                # First try inference mode (no targets)
                outputs = model(images)
                if isinstance(outputs, dict):
                    # Handle FarSeg output format
                    logits = outputs.get('seg', outputs.get('cls', outputs))
                else:
                    logits = outputs
            except:
                # Fallback: try with targets for models that require them
                outputs = model(images, targets)
                if isinstance(outputs, dict):
                    # During training/eval with targets, look for prediction outputs
                    if 'pred' in outputs:
                        logits = outputs['pred']
                    elif 'seg' in outputs:
                        logits = outputs['seg']
                    elif 'cls' in outputs:
                        logits = outputs['cls']
                    else:
                        # If it's a loss dict, try inference mode
                        outputs = model(images)
                        logits = outputs
                else:
                    logits = outputs
            
            predictions = torch.argmax(logits, dim=1)
            ground_truth = targets['cls']
            
            # Store for metrics calculation
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(ground_truth.cpu())
            all_images.append(images.cpu())
            
            sample_count += images.size(0)
            
            # Update progress
            pbar.set_postfix({'Samples': sample_count})
    
    pbar.close()
    
    # Concatenate all predictions and ground truth
    all_predictions = torch.cat(all_predictions, dim=0)
    all_ground_truth = torch.cat(all_ground_truth, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    print(f"\n🔍 Computing evaluation metrics...")
    
    # Calculate pixel-wise accuracy
    valid_mask = (all_ground_truth != 255)  # Ignore index
    valid_predictions = all_predictions[valid_mask]
    valid_ground_truth = all_ground_truth[valid_mask]
    
    pixel_accuracy = (valid_predictions == valid_ground_truth).float().mean().item()
    
    # Calculate IoU for each class
    class_ious = []
    for i in range(len(all_predictions)):
        ious = calculate_iou(all_predictions[i], all_ground_truth[i], num_classes)
        class_ious.append(ious)
    
    # Average IoU across all samples
    class_ious = np.array(class_ious)
    mean_iou_per_class = np.nanmean(class_ious, axis=0)
    # Calculate mean IoU only for classes that have samples
    valid_class_ious = mean_iou_per_class[~np.isnan(mean_iou_per_class)]
    mean_iou = np.mean(valid_class_ious) if len(valid_class_ious) > 0 else 0.0
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        valid_ground_truth.numpy(), 
        valid_predictions.numpy(),
        labels=list(range(num_classes))
    )
    
    # Classification report
    class_report = classification_report(
        valid_ground_truth.numpy(),
        valid_predictions.numpy(),
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0
    )
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_ground_truth.numpy(),
        valid_predictions.numpy(),
        labels=list(range(num_classes)),
        average=None,
        zero_division=0
    )
    
    evaluation_time = time.time() - start_time
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS")
    print(f"=" * 60)
    print(f"Dataset: {os.path.basename(config_path).replace('farseg_', '').replace('.py', '')}")
    print(f"Model: FarSeg")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"Test samples: {sample_count}")
    print(f"Evaluation time: {evaluation_time:.1f}s")
    print(f"")
    print(f"Overall Metrics:")
    print(f"  Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"")
    print(f"Per-Class Metrics:")
    for class_id in range(num_classes):
        iou_val = mean_iou_per_class[class_id]
        iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A (no samples)"
        print(f"  Class {class_id}:")
        print(f"    IoU: {iou_str}")
        print(f"    Precision: {precision[class_id]:.4f}")
        print(f"    Recall: {recall[class_id]:.4f}")
        print(f"    F1-Score: {f1[class_id]:.4f}")
        print(f"    Support: {support[class_id]}")
    
    # Save detailed results
    results = {
        'dataset': os.path.basename(config_path).replace('farseg_', '').replace('.py', ''),
        'model': 'FarSeg',
        'checkpoint': os.path.basename(checkpoint_path),
        'trained_iterations': trained_iters,
        'test_samples': sample_count,
        'evaluation_time': evaluation_time,
        'metrics': {
            'pixel_accuracy': pixel_accuracy,
            'mean_iou': mean_iou,
            'per_class_iou': mean_iou_per_class.tolist(),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
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
        f.write(f"FarSeg Evaluation Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"Checkpoint: {results['checkpoint']}\n")
        f.write(f"Test samples: {results['test_samples']}\n")
        f.write(f"Evaluation time: {results['evaluation_time']:.1f}s\n")
        f.write(f"\nOverall Metrics:\n")
        f.write(f"  Pixel Accuracy: {results['metrics']['pixel_accuracy']:.4f}\n")
        f.write(f"  Mean IoU: {results['metrics']['mean_iou']:.4f}\n")
        f.write(f"\nPer-Class Metrics:\n")
        for class_id in range(num_classes):
            f.write(f"  Class {class_id}:\n")
            f.write(f"    IoU: {results['metrics']['per_class_iou'][class_id]:.4f}\n")
            f.write(f"    Precision: {results['metrics']['per_class_precision'][class_id]:.4f}\n")
            f.write(f"    Recall: {results['metrics']['per_class_recall'][class_id]:.4f}\n")
            f.write(f"    F1-Score: {results['metrics']['per_class_f1'][class_id]:.4f}\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    confusion_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save sample predictions
    samples_dir = os.path.join(output_dir, 'prediction_samples')
    save_prediction_samples(all_images, all_predictions, all_ground_truth, samples_dir)
    
    print(f"\n💾 Results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  Text: {results_txt}")
    print(f"  Confusion matrix: {confusion_matrix_file}")
    print(f"  Sample predictions: {samples_dir}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate FarSeg model')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--model_dir', required=True, help='Model directory containing checkpoints')
    parser.add_argument('--output_dir', help='Output directory for evaluation results (default: model_dir/evaluation)')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to evaluate (default: latest)')
    parser.add_argument('--gpu_ids', default="0", help='GPU IDs (e.g., "0,1,2,3")')
    
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
        checkpoint_path=args.checkpoint
    )

if __name__ == '__main__':
    main()
