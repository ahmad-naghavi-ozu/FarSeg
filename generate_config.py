#!/usr/bin/env python3
"""
Generic configuration generator for FarSeg with standardized dataset format.

Usage:
    python generate_config.py --dataset_name my_dataset --num_classes 5 --data_root /path/to/dataset
"""

import argparse
import os
import json
from typing import List, Dict, Any


def generate_config(dataset_name: str, 
                   num_classes: int,
                   data_root: str,
                   class_values: List[int] = None,
                   patch_size: int = 896,
                   stride: int = 512,
                   batch_size_train: int = 4,
                   batch_size_test: int = 1,
                   base_lr: float = 0.007,
                   max_iters: int = 60000,
                   use_train_valid_fusion: bool = False,
                   model_type: str = "farseg") -> Dict[str, Any]:
    """
    Generate a generic configuration for FarSeg model.
    
    Args:
        dataset_name: Name of the dataset
        num_classes: Number of classes (including background)
        data_root: Root path to the dataset
        class_values: List of class values in masks (if None, assumes 0,1,2,...)
        patch_size: Size of input patches
        stride: Stride for patch extraction
        batch_size_train: Training batch size
        batch_size_test: Test batch size
        base_lr: Base learning rate
        max_iters: Maximum training iterations
        use_train_valid_fusion: If True, combine train and valid sets for training (manuscript strategy)
                                Testing ALWAYS uses test split regardless of this flag
    
    Returns:
        Configuration dictionary
    """
    
    if class_values is None:
        class_values = list(range(num_classes))
    
    # Paths according to standardized format
    train_image_dir = os.path.join(data_root, dataset_name, 'train', 'rgb')
    train_mask_dir = os.path.join(data_root, dataset_name, 'train', 'sem')
    val_image_dir = os.path.join(data_root, dataset_name, 'valid', 'rgb')
    val_mask_dir = os.path.join(data_root, dataset_name, 'valid', 'sem')
    test_image_dir = os.path.join(data_root, dataset_name, 'test', 'rgb')
    test_mask_dir = os.path.join(data_root, dataset_name, 'test', 'sem')
    
    # For train+valid fusion, we'll use a special multi-directory configuration
    if use_train_valid_fusion:
        # Training will use both train and valid directories (manuscript strategy)
        train_image_dirs = [train_image_dir, val_image_dir]
        train_mask_dirs = [train_mask_dir, val_mask_dir]
        # Testing ALWAYS uses the test split only
        test_image_dir = test_image_dir
        test_mask_dir = test_mask_dir
    else:
        # Normal configuration - train on train split only
        train_image_dirs = train_image_dir
        train_mask_dirs = train_mask_dir
        # Testing ALWAYS uses the test split only
        test_image_dir = test_image_dir
        test_mask_dir = test_mask_dir
    
    # Map model type to configuration model type
    if model_type.lower() == "farsegpp":
        config_model_type = "FarSegPP"
    else:
        config_model_type = "FarSeg"
    
    config = {
        'model': {
            'type': config_model_type,
            'params': {
                'resnet_encoder': {
                    'resnet_type': 'resnet50',
                    'include_conv5': True,
                    'batchnorm_trainable': True,
                    'pretrained': True,
                    'freeze_at': 0,
                    'output_stride': 32,
                    'with_cp': (False, False, False, False),
                    'stem3_3x3': False,
                },
                'fpn': {
                    'in_channels_list': (256, 512, 1024, 2048),
                    'out_channels': 256,
                    'conv_block': 'fpn.default_conv_block',
                    'top_blocks': None,
                },
                'scene_relation': {
                    'in_channels': 2048,
                    'channel_list': (256, 256, 256, 256),
                    'out_channels': 256,
                    'scale_aware_proj': True,
                },
                'decoder': {
                    'in_channels': 256,
                    'out_channels': 128,
                    'in_feat_output_strides': (4, 8, 16, 32),
                    'out_feat_output_stride': 4,
                    'norm_fn': 'nn.BatchNorm2d',
                    'num_groups_gn': None
                },
                'num_classes': num_classes,
                'loss': {
                    'cls_weight': 1.0,
                    'ignore_index': 255,
                },
                'annealing_softmax_focalloss': {
                    'gamma': 2.0,
                    'max_step': max_iters,
                    'annealing_type': 'cosine'
                },
            }
        },
        'data': {
            'train': {
                'type': 'GenericFusionSegmentationDataLoader' if use_train_valid_fusion else 'GenericSegmentationDataLoader',
                'params': {
                    'image_dir': train_image_dirs,
                    'mask_dir': train_mask_dirs,
                    'dataset_name': dataset_name,
                    'num_classes': num_classes,
                    'class_values': class_values,
                    'patch_config': {
                        'patch_size': patch_size,
                        'stride': stride,
                    },
                    'transforms': 'TRAIN_TRANSFORMS',  # Will be replaced in actual config
                    'batch_size': batch_size_train,
                    'num_workers': 8,
                    'training': True
                },
            },
            'test': {
                'type': 'GenericSegmentationDataLoader',
                'params': {
                    'image_dir': test_image_dir,
                    'mask_dir': test_mask_dir,
                    'dataset_name': dataset_name,
                    'num_classes': num_classes,
                    'class_values': class_values,
                    'patch_config': {
                        'patch_size': patch_size,
                        'stride': stride,
                    },
                    'transforms': 'TEST_TRANSFORMS',  # Will be replaced in actual config
                    'batch_size': batch_size_test,
                    'num_workers': 0,
                    'training': False
                },
            },
        },
        'optimizer': {
            'type': 'sgd',
            'params': {
                'momentum': 0.9,
                'weight_decay': 0.0001
            },
            'grad_clip': {
                'max_norm': 35,
                'norm_type': 2,
            }
        },
        'learning_rate': {
            'type': 'poly',
            'params': {
                'base_lr': base_lr,
                'power': 0.9,
                'max_iters': max_iters,
            }
        },
        'train': {
            'forward_times': 1,
            'num_iters': max_iters,
            'eval_per_epoch': False,
            'summary_grads': False,
            'summary_weights': False,
            'distributed': True,
            'apex_sync_bn': True,
            'sync_bn': False,
            'eval_after_train': False,
            'log_interval_step': 50,
        },
        'test': {
        },
    }
    
    return config


def create_python_config_file(config: Dict[str, Any], 
                            output_path: str, 
                            dataset_name: str):
    """Create a Python configuration file from the config dictionary."""
    
    model_type = config['model']['type']
    
    config_template = '''import torch.nn as nn
from simplecv.module import fpn

from data.generic_dataset import GenericRemoveColorMap
from simplecv.api.preprocess import segm
from simplecv.api.preprocess import comm

# Configuration for {dataset_name} dataset
config = {{
    "model": {{
        "type": "{model_type}",
        "params": {{
            "resnet_encoder": {{
                "resnet_type": "resnet50",
                "include_conv5": True,
                "batchnorm_trainable": True,
                "pretrained": True,
                "freeze_at": 0,
                "output_stride": 32,
                "with_cp": (False, False, False, False),
                "stem3_3x3": False,
            }},
            "fpn": {{
                "in_channels_list": (256, 512, 1024, 2048),
                "out_channels": 256,
                "conv_block": fpn.default_conv_block,
                "top_blocks": None,
            }},
            "scene_relation": {{
                "in_channels": 2048,
                "channel_list": (256, 256, 256, 256),
                "out_channels": 256,
                "scale_aware_proj": True,
            }},
            "decoder": {{
                "in_channels": 256,
                "out_channels": 128,
                "in_feat_output_strides": (4, 8, 16, 32),
                "out_feat_output_stride": 4,
                "norm_fn": nn.BatchNorm2d,
                "num_groups_gn": None
            }},
            "num_classes": {num_classes},
            "loss": {{
                "cls_weight": 1.0,
                "ignore_index": 255,
            }},
            "annealing_softmax_focalloss": {{
                "gamma": 2.0,
                "max_step": {max_iters},
                "annealing_type": "cosine"
            }},
        }}
    }},
    "data": {{
        "train": {{
            "type": "{train_dataloader_type}",
            "params": {{
                "image_dir": "{train_image_dir}",
                "mask_dir": "{train_mask_dir}",
                "dataset_name": "{dataset_name}",
                "num_classes": {num_classes},
                "class_values": {class_values},
                "patch_config": {{
                    "patch_size": {patch_size},
                    "stride": {stride},
                }},
                "transforms": [
                    GenericRemoveColorMap(class_values={class_values}, num_classes={num_classes}),
                    segm.RandomHorizontalFlip(0.5),
                    segm.RandomVerticalFlip(0.5),
                    segm.RandomRotate90K((0, 1, 2, 3)),
                    segm.FixedPad(({patch_size}, {patch_size}), 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                "batch_size": {batch_size_train},
                "num_workers": 8,
                "training": True
            }},
        }},
        "test": {{
            "type": "GenericSegmentationDataLoader",
            "params": {{
                "image_dir": "{val_image_dir}",
                "mask_dir": "{val_mask_dir}",
                "dataset_name": "{dataset_name}",
                "num_classes": {num_classes},
                "class_values": {class_values},
                "patch_config": {{
                    "patch_size": {patch_size},
                    "stride": {stride},
                }},
                "transforms": [
                    GenericRemoveColorMap(class_values={class_values}, num_classes={num_classes}),
                    segm.DivisiblePad(32, 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                "batch_size": {batch_size_test},
                "num_workers": 0,
                "training": False
            }},
        }},
    }},
    "optimizer": {{
        "type": "sgd",
        "params": {{
            "momentum": 0.9,
            "weight_decay": 0.0001
        }},
        "grad_clip": {{
            "max_norm": 35,
            "norm_type": 2,
        }}
    }},
    "learning_rate": {{
        "type": "poly",
        "params": {{
            "base_lr": {base_lr},
            "power": 0.9,
            "max_iters": {max_iters},
        }}
    }},
    "train": {{
        "forward_times": 1,
        "num_iters": {max_iters},
        "eval_per_epoch": False,
        "summary_grads": False,
        "summary_weights": False,
        "distributed": True,
        "apex_sync_bn": True,
        "sync_bn": False,
        "eval_after_train": False,
        "log_interval_step": 50,
    }},
    "test": {{
    }},
}}
'''

    # Extract values from config
    train_params = config['data']['train']['params']
    test_params = config['data']['test']['params']
    model_params = config['model']['params']
    
    formatted_config = config_template.format(
        dataset_name=dataset_name,
        model_type=model_type,
        num_classes=model_params['num_classes'],
        max_iters=config['learning_rate']['params']['max_iters'],
        train_dataloader_type=config['data']['train']['type'],
        train_image_dir=train_params['image_dir'],
        train_mask_dir=train_params['mask_dir'],
        val_image_dir=test_params['image_dir'],
        val_mask_dir=test_params['mask_dir'],
        class_values=train_params['class_values'],
        patch_size=train_params['patch_config']['patch_size'],
        stride=train_params['patch_config']['stride'],
        batch_size_train=train_params['batch_size'],
        batch_size_test=test_params['batch_size'],
        base_lr=config['learning_rate']['params']['base_lr']
    )
    
    with open(output_path, 'w') as f:
        f.write(formatted_config)


def main():
    parser = argparse.ArgumentParser(description='Generate FarSeg configuration for generic datasets')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--model_type', type=str, default='farseg',
                       choices=['farseg', 'farsegpp'],
                       help='Model type: farseg or farsegpp (default: farseg)')
    parser.add_argument('--num_classes', type=int, required=True,
                       help='Number of classes including background')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing the dataset')
    parser.add_argument('--class_values', type=str, default=None,
                       help='Comma-separated list of class values (e.g., "0,1,2,3")')
    parser.add_argument('--patch_size', type=int, default=896,
                       help='Input patch size (default: 896)')
    parser.add_argument('--stride', type=int, default=512,
                       help='Patch extraction stride (default: 512)')
    parser.add_argument('--batch_size_train', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--batch_size_test', type=int, default=1,
                       help='Test batch size (default: 1)')
    parser.add_argument('--base_lr', type=float, default=0.007,
                       help='Base learning rate (default: 0.007)')
    parser.add_argument('--max_iters', type=int, default=60000,
                       help='Maximum training iterations (default: 60000)')
    parser.add_argument('--output_dir', type=str, default='./configs',
                       help='Output directory for config files (default: ./configs)')
    parser.add_argument('--use_train_valid_fusion', action='store_true',
                       help='Combine train and valid sets for training (no validation split)')
    
    args = parser.parse_args()
    
    # Parse class values
    if args.class_values:
        class_values = [int(x.strip()) for x in args.class_values.split(',')]
    else:
        class_values = list(range(args.num_classes))
    
    # Generate configuration
    config = generate_config(
        dataset_name=args.dataset_name,
        num_classes=args.num_classes,
        data_root=args.data_root,
        class_values=class_values,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        base_lr=args.base_lr,
        max_iters=args.max_iters,
        use_train_valid_fusion=args.use_train_valid_fusion,
        model_type=args.model_type
    )
    
    # Create output directory (output_dir is already model-specific: model_type/dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save Python config file directly in the output directory
    config_py_path = os.path.join(args.output_dir, f'farseg_{args.dataset_name}.py')
    create_python_config_file(config, config_py_path, args.dataset_name)
    
    # Save JSON config for reference
    config_json_path = os.path.join(args.output_dir, f'config_{args.dataset_name}.json')
    with open(config_json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration files generated:")
    print(f"  Python config: {config_py_path}")
    print(f"  JSON config: {config_json_path}")
    print(f"\nTo use this configuration:")
    print(f"  python apex_train.py --config_path {config_py_path}")


if __name__ == '__main__':
    main()
