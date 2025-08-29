import torch.nn as nn
from simplecv.module import fpn

from data.generic_dataset import GenericRemoveColorMap
from simplecv.api.preprocess import segm
from simplecv.api.preprocess import comm

# Configuration for DFC2023S dataset
config = {
    "model": {
        "type": "FarSeg",
        "params": {
            "resnet_encoder": {
                "resnet_type": "resnet50",
                "include_conv5": True,
                "batchnorm_trainable": True,
                "pretrained": True,
                "freeze_at": 0,
                "output_stride": 32,
                "with_cp": (False, False, False, False),
                "stem3_3x3": False,
            },
            "fpn": {
                "in_channels_list": (256, 512, 1024, 2048),
                "out_channels": 256,
                "conv_block": fpn.default_conv_block,
                "top_blocks": None,
            },
            "scene_relation": {
                "in_channels": 2048,
                "channel_list": (256, 256, 256, 256),
                "out_channels": 256,
                "scale_aware_proj": True,
            },
            "decoder": {
                "in_channels": 256,
                "out_channels": 128,
                "in_feat_output_strides": (4, 8, 16, 32),
                "out_feat_output_stride": 4,
                "norm_fn": nn.BatchNorm2d,
                "num_groups_gn": None
            },
            "num_classes": 2,
            "loss": {
                "cls_weight": 1.0,
                "ignore_index": 255,
            },
            "annealing_softmax_focalloss": {
                "gamma": 2.0,
                "max_step": 100,
                "annealing_type": "cosine"
            },
        }
    },
    "data": {
        "train": {
            "type": "GenericFusionSegmentationDataLoader",
            "params": {
                "image_dir": "['/home/asfand/Ahmad/datasets/DFC2023S/train/rgb', '/home/asfand/Ahmad/datasets/DFC2023S/valid/rgb']",
                "mask_dir": "['/home/asfand/Ahmad/datasets/DFC2023S/train/sem', '/home/asfand/Ahmad/datasets/DFC2023S/valid/sem']",
                "dataset_name": "DFC2023S",
                "num_classes": 2,
                "class_values": [0, 1],
                "patch_config": {
                    "patch_size": 896,
                    "stride": 512,
                },
                "transforms": [
                    GenericRemoveColorMap(class_values=[0, 1], num_classes=2),
                    segm.RandomHorizontalFlip(0.5),
                    segm.RandomVerticalFlip(0.5),
                    segm.RandomRotate90K((0, 1, 2, 3)),
                    segm.FixedPad((896, 896), 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                "batch_size": 8,
                "num_workers": 8,
                "training": True
            },
        },
        "test": {
            "type": "GenericSegmentationDataLoader",
            "params": {
                "image_dir": "/home/asfand/Ahmad/datasets/DFC2023S/test/rgb",
                "mask_dir": "/home/asfand/Ahmad/datasets/DFC2023S/test/sem",
                "dataset_name": "DFC2023S",
                "num_classes": 2,
                "class_values": [0, 1],
                "patch_config": {
                    "patch_size": 896,
                    "stride": 512,
                },
                "transforms": [
                    GenericRemoveColorMap(class_values=[0, 1], num_classes=2),
                    segm.DivisiblePad(32, 255),
                    segm.ToTensor(True),
                    comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
                ],
                "batch_size": 1,
                "num_workers": 0,
                "training": False
            },
        },
    },
    "optimizer": {
        "type": "sgd",
        "params": {
            "momentum": 0.9,
            "weight_decay": 0.0001
        },
        "grad_clip": {
            "max_norm": 35,
            "norm_type": 2,
        }
    },
    "learning_rate": {
        "type": "poly",
        "params": {
            "base_lr": 0.007,
            "power": 0.9,
            "max_iters": 100,
        }
    },
    "train": {
        "forward_times": 1,
        "num_iters": 100,
        "eval_per_epoch": False,
        "summary_grads": False,
        "summary_weights": False,
        "distributed": True,
        "apex_sync_bn": True,
        "sync_bn": False,
        "eval_after_train": False,
        "log_interval_step": 50,
    },
    "test": {
    },
}
