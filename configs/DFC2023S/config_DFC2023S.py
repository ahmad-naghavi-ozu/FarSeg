import torch.nn as nn
from simplecv.module import fpn
from simplecv.api.preprocess import segm
from simplecv.api.preprocess import comm

config = dict(
    model=dict(
        type='FarSeg',
        params=dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=fpn.default_conv_block,
                top_blocks=None,
            ),
            scene_relation=dict(
                in_channels=2048,
                channel_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=2,
            loss=dict(
                cls_weight=1.0,
                ignore_index=255,
            ),
        )
    ),
    data=dict(
        train=dict(
            type='GenericSegmentationDataset',
            params=dict(
                image_dir='../../datasets/DFC2023S/train/rgb',
                mask_dir='../../datasets/DFC2023S/train/sem',
                transforms=comm.Compose([
                    segm.RandomHorizontalFlip(prob=0.5),
                    segm.ToTensor(),
                    comm.THMeanStdNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
        ),
        test=dict(
            type='GenericSegmentationDataset',
            params=dict(
                image_dir='../../datasets/DFC2023S/valid/rgb',
                mask_dir='../../datasets/DFC2023S/valid/sem',
                transforms=comm.Compose([
                    segm.ToTensor(),
                    comm.THMeanStdNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            )
        )
    ),
    optimizer=dict(
        type='SGD',
        params=dict(
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001
        )
    ),
    learning_rate=dict(
        type='PolynomialLR',
        params=dict(
            max_epoch=60,
            power=0.9
        )
    ),
    train=dict(
        epochs=60,
        batch_size=8,
        num_workers=4,
        save_interval_epochs=10,
        eval_interval_epochs=5,
        log_interval_steps=100,
        save_ckpt_interval_epoch=10,
        eval_during_train=True,
        eval_after_train=True
    ),
    test=dict(
        batch_size=8,
        num_workers=4
    )
)
