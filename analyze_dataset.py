#!/usr/bin/env python3
"""
Dataset analysis utility for generic segmentation datasets.

This script analyzes the class distribution in semantic segmentation    # Backgrou    # Configuration suggestions
    print("\n🔧 SUGGESTED CONFIGURATION:")
    print(f"   --num_classes {dataset_info['num_classes']}")
    print(f"   --class_values \"{','.join(map(str, dataset_info['unique_class_values']))}\"")
    
    if dataset_info['num_classes'] <= 5:
        print("   Consider increasing patch_size for small number of classes")
    elif dataset_info['num_classes'] > 15:
        print("   Consider using larger model capacity for many classes")detection
    if 0 in class_dist and class_dist[0]['percentage'] > 80:
        print("\n🏞️  Large background class detected:")
        print(f"   Background (class 0): {class_dist[0]['percentage']:.2f}%")
        print("   FarSeg's foreground-aware optimization will help with this")
    
    # Configuration suggestions
    print("\n🔧 SUGGESTED CONFIGURATION:")
    print(f"   --num_classes {dataset_info['num_classes']}")
    print(f"   --class_values \"{','.join(map(str, dataset_info['unique_class_values']))}\"\") provides statistics to help configure the model properly.

Usage:
    python analyze_dataset.py --dataset_path /path/to/dataset --dataset_name my_dataset
"""

import argparse
import os
import glob
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
import json


def find_images_with_extensions(directory, extensions=None):
    """Find all images in directory with specified extensions"""
    if extensions is None:
        extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']
    
    image_files = []
    for ext in extensions:
        # Check both lowercase and uppercase
        pattern_lower = os.path.join(directory, f'*{ext.lower()}')
        pattern_upper = os.path.join(directory, f'*{ext.upper()}')
        image_files.extend(glob.glob(pattern_lower))
        image_files.extend(glob.glob(pattern_upper))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort


def analyze_masks(mask_dir, mask_extension='.png'):
    """
    Analyze semantic segmentation masks to extract class statistics.
    
    Args:
        mask_dir: Directory containing mask files
        mask_extension: File extension for mask files (if 'auto', will search for all supported formats)
    
    Returns:
        Dictionary containing analysis results
    """
    if mask_extension == 'auto':
        # Auto-detect mask files with various extensions
        mask_files = find_images_with_extensions(mask_dir)
    else:
        # Use specific extension
        mask_pattern = os.path.join(mask_dir, f'*{mask_extension}')
        mask_files = glob.glob(mask_pattern)
        # Also try uppercase
        mask_pattern_upper = os.path.join(mask_dir, f'*{mask_extension.upper()}')
        mask_files.extend(glob.glob(mask_pattern_upper))
        mask_files = sorted(list(set(mask_files)))  # Remove duplicates and sort
    
    if not mask_files:
        print(f"No mask files found in {mask_dir} with extension {mask_extension}")
        return None
    
    print(f"Found {len(mask_files)} mask files")
    
    overall_class_counts = Counter()
    per_image_stats = []
    all_unique_values = set()
    
    for i, mask_file in enumerate(mask_files):
        if i % 100 == 0:
            print(f"Processing {i}/{len(mask_files)} files...")
        
        try:
            # Load mask
            mask = imread(mask_file)
            
            # Handle different mask formats
            if len(mask.shape) == 3:
                # If RGB mask, convert to single channel
                if mask.shape[2] == 3:
                    # Take the first channel or convert based on your specific format
                    mask = mask[:, :, 0]
                else:
                    mask = mask.squeeze()
            
            # Get unique values and their counts
            unique_values, counts = np.unique(mask, return_counts=True)
            all_unique_values.update(unique_values)
            
            # Update overall counts
            for val, count in zip(unique_values, counts):
                overall_class_counts[val] += count
            
            # Store per-image statistics
            per_image_stats.append({
                'filename': os.path.basename(mask_file),
                'unique_classes': unique_values.tolist(),
                'class_counts': counts.tolist(),
                'total_pixels': mask.size,
                'num_classes': len(unique_values)
            })
            
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
            continue
    
    # Calculate statistics
    total_pixels = sum(overall_class_counts.values())
    sorted_classes = sorted(all_unique_values)
    
    analysis_results = {
        'dataset_info': {
            'num_mask_files': len(mask_files),
            'total_pixels_analyzed': total_pixels,
            'unique_class_values': sorted_classes,
            'num_classes': len(sorted_classes)
        },
        'class_distribution': {
            int(cls): {
                'pixel_count': int(overall_class_counts[cls]),
                'percentage': (overall_class_counts[cls] / total_pixels) * 100,
                'num_images_containing': sum(1 for img_stat in per_image_stats 
                                           if cls in img_stat['unique_classes'])
            }
            for cls in sorted_classes
        },
        'per_image_stats': per_image_stats[:10]  # Store first 10 for reference
    }
    
    return analysis_results


def print_analysis_summary(analysis_results):
    """Print a summary of the analysis results."""
    if not analysis_results:
        return
    
    dataset_info = analysis_results['dataset_info']
    class_dist = analysis_results['class_distribution']
    
    print("\n" + "="*50)
    print("DATASET ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Number of mask files: {dataset_info['num_mask_files']}")
    print(f"Total pixels analyzed: {dataset_info['total_pixels_analyzed']:,}")
    print(f"Number of unique classes: {dataset_info['num_classes']}")
    print(f"Class values found: {dataset_info['unique_class_values']}")
    
    print("\n" + "-"*50)
    print("CLASS DISTRIBUTION")
    print("-"*50)
    print(f"{'Class':<8} {'Pixels':<12} {'Percentage':<12} {'In # Images':<12}")
    print("-"*50)
    
    for cls_val in sorted(class_dist.keys()):
        cls_info = class_dist[cls_val]
        print(f"{cls_val:<8} {cls_info['pixel_count']:<12,} "
              f"{cls_info['percentage']:<12.2f} {cls_info['num_images_containing']:<12}")
    
    # Recommendations
    print("\n" + "-"*50)
    print("RECOMMENDATIONS")
    print("-"*50)
    
    # Check for class imbalance
    percentages = [info['percentage'] for info in class_dist.values()]
    max_percentage = max(percentages)
    min_percentage = min(percentages)
    imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')
    
    if imbalance_ratio > 100:
        print("⚠️  High class imbalance detected!")
        print(f"   Most frequent class: {max_percentage:.2f}%")
        print(f"   Least frequent class: {min_percentage:.2f}%")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        print("   Consider using class weighting or focal loss")
    
    # Background class detection
    if 0 in class_dist and class_dist[0]['percentage'] > 80:
        print("\n🏞️  Large background class detected:")
        print(f"   Background (class 0): {class_dist[0]['percentage']:.2f}%")
        print("   FarSeg's foreground-aware optimization will help with this")
    
    # Configuration suggestions
    print("\n🔧 SUGGESTED CONFIGURATION:")
    print(f"   --num_classes {dataset_info['num_classes']}")
    print(f"   --class_values \"{','.join(map(str, dataset_info['unique_class_values']))}\"")
    
    if dataset_info['num_classes'] <= 5:
        print("   Consider increasing patch_size for small number of classes")
    elif dataset_info['num_classes'] > 15:
        print("   Consider using larger model capacity for many classes")


def visualize_class_distribution(analysis_results, output_dir=None):
    """Create visualizations of the class distribution."""
    if not analysis_results:
        return
    
    class_dist = analysis_results['class_distribution']
    
    # Prepare data for plotting
    class_values = list(class_dist.keys())
    percentages = [class_dist[cls]['percentage'] for cls in class_values]
    pixel_counts = [class_dist[cls]['pixel_count'] for cls in class_values]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Percentage distribution (bar chart)
    bars1 = ax1.bar(range(len(class_values)), percentages)
    ax1.set_xlabel('Class Value')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Class Distribution (Percentage)')
    ax1.set_xticks(range(len(class_values)))
    ax1.set_xticklabels(class_values)
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars1, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    # Pixel count distribution (log scale)
    bars2 = ax2.bar(range(len(class_values)), pixel_counts)
    ax2.set_xlabel('Class Value')
    ax2.set_ylabel('Pixel Count (log scale)')
    ax2.set_title('Class Distribution (Pixel Count)')
    ax2.set_xticks(range(len(class_values)))
    ax2.set_xticklabels(class_values)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'class_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Class distribution plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze segmentation dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset root directory')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset subdirectory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to analyze (default: train)')
    parser.add_argument('--mask_extension', type=str, default='auto',
                       help='File extension for mask files (default: auto - detects .png, .tif, .tiff, .jpg, .jpeg, .bmp, .gif)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis results and plots')
    parser.add_argument('--save_json', action='store_true',
                       help='Save detailed analysis results as JSON')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and show class distribution plots')
    
    args = parser.parse_args()
    
    # Construct path to semantic masks
    mask_dir = os.path.join(args.dataset_path, args.dataset_name, args.split, 'sem')
    
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory not found: {mask_dir}")
        print("Expected directory structure:")
        print(f"  {args.dataset_path}/")
        print(f"    {args.dataset_name}/")
        print(f"      {args.split}/")
        print("        sem/  <- mask files should be here")
        return
    
    print(f"Analyzing dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print(f"Mask directory: {mask_dir}")
    
    # Perform analysis
    analysis_results = analyze_masks(mask_dir, args.mask_extension)
    
    if analysis_results:
        # Print summary
        print_analysis_summary(analysis_results)
        
        # Save detailed results if requested
        if args.save_json and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            json_path = os.path.join(args.output_dir, f'{args.dataset_name}_analysis.json')
            with open(json_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            print(f"\n💾 Detailed analysis saved to: {json_path}")
        
        # Generate plots if requested
        if args.plot:
            try:
                visualize_class_distribution(analysis_results, args.output_dir)
            except ImportError:
                print("\n⚠️  Matplotlib not available for plotting")
    else:
        print("Analysis failed - no valid mask files found")


if __name__ == '__main__':
    main()
