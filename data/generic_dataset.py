import glob
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from simplecv import registry
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
from simplecv.core.config import AttrDict
from simplecv.data import distributed
from simplecv.util import viz
from skimage.io import imread
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.patch_base import PatchBasedDataset

DEFAULT_PATCH_CONFIG = dict(
    patch_size=896,
    stride=512,
)


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


class GenericRemoveColorMap(object):
    """Generic color map remover that works with different color mappings"""
    def __init__(self, class_values=None, num_classes=None):
        """
        Args:
            class_values: List of class values in the segmentation masks (e.g., [0, 1, 2, 3, ...])
            num_classes: Number of classes including background
        """
        super(GenericRemoveColorMap, self).__init__()
        if class_values is None:
            # Default: assume classes are 0, 1, 2, ..., num_classes-1
            self.class_values = list(range(num_classes)) if num_classes else [0, 1]
        else:
            self.class_values = class_values
        
        self.num_classes = len(self.class_values)

    def __call__(self, image, mask):
        if isinstance(mask, Image.Image):
            mask = np.array(mask, copy=False)
        
        # If mask is RGB color format, convert to grayscale labels
        if len(mask.shape) == 3:
            # Assume it's a color mask that needs conversion
            # For now, just take the first channel or convert properly
            if mask.shape[2] == 3:
                # Simple conversion - you might need to adjust this based on your color scheme
                mask = mask[:, :, 0]  # Take red channel as example
        
        # Ensure mask values are within expected range
        unique_values = np.unique(mask)
        print(f"Unique mask values found: {unique_values}")
        
        return image, Image.fromarray(mask.astype(np.uint8, copy=False))


class GenericSegmentationDataset(PatchBasedDataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 patch_config=DEFAULT_PATCH_CONFIG,
                 transforms=None,
                 image_extension='.png',
                 mask_extension='.png'):
        """
        Generic segmentation dataset for the standardized format:
        dataset/
        ├── train/rgb/ (images)
        └── train/sem/ (masks)
        
        Args:
            image_dir: Path to RGB images (e.g., 'dataset/train/rgb')
            mask_dir: Path to semantic masks (e.g., 'dataset/train/sem')
            patch_config: Configuration for patch-based processing
            transforms: List of transforms to apply
            image_extension: File extension for images
            mask_extension: File extension for masks
        """
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        super(GenericSegmentationDataset, self).__init__(image_dir, mask_dir, patch_config, transforms=transforms)

    def generate_path_pair(self):
        """Generate pairs of image and mask paths"""
        image_pattern = os.path.join(self.image_dir, f'*{self.image_extension}')
        image_path_list = glob.glob(image_pattern)
        
        mask_path_list = []
        for img_path in image_path_list:
            img_name = os.path.basename(img_path)
            # Remove extension and add mask extension
            mask_name = os.path.splitext(img_name)[0] + self.mask_extension
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask_path_list.append(mask_path)

        return zip(image_path_list, mask_path_list)

    def show_image_mask(self, idx, mask_on=True, ax=None):
        img_tensor, blob = self[idx]
        img = img_tensor.numpy()
        mask = blob['cls'].numpy()
        if mask_on:
            img = np.where(mask.sum() == 0, img, img * 0.5 + (1 - 0.5) * mask)

        viz.plot_image(img, ax)

    def __getitem__(self, idx):
        img_tensor, y = super(GenericSegmentationDataset, self).__getitem__(idx)
        mask_tensor = y['cls']
        # rm background
        multi_cls_label = torch.unique(mask_tensor)
        # start from 0, exclude background (0) and ignore label (255)
        fg_cls_label = multi_cls_label[(multi_cls_label > 0) & (multi_cls_label != 255)] - 1
        
        # Get number of classes dynamically
        max_class = torch.max(mask_tensor[mask_tensor != 255]).item() if torch.any(mask_tensor != 255) else 0
        num_classes = max_class  # excluding background
        
        if len(fg_cls_label) > 0 and num_classes > 0:
            y['fg_cls_label'] = F.one_hot(fg_cls_label.long(), num_classes=num_classes).sum(dim=0)
        else:
            y['fg_cls_label'] = torch.zeros(num_classes, dtype=torch.float32)
        
        return img_tensor, y


@registry.DATALOADER.register('GenericSegmentationDataLoader')
class GenericSegmentationDataLoader(DataLoader):
    def __init__(self, config):
        self.config = AttrDict()
        self.set_default()
        self.config.update(config)

        # Get dataset info
        dataset_name = self.config.get('dataset_name', 'generic')
        class_values = self.config.get('class_values', None)
        num_classes = self.config.get('num_classes', 2)

        # Setup transforms with appropriate color map remover
        transforms = self.config.transforms.copy()
        if transforms and isinstance(transforms[0], type) and hasattr(transforms[0], '__name__'):
            # Replace the first transform if it's a color map remover
            if 'RemoveColorMap' in str(transforms[0]):
                transforms[0] = GenericRemoveColorMap(class_values=class_values, num_classes=num_classes)

        dataset = GenericSegmentationDataset(
            self.config.image_dir,
            self.config.mask_dir,
            self.config.patch_config,
            transforms,
            self.config.get('image_extension', '.png'),
            self.config.get('mask_extension', '.png')
        )

        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

        super(GenericSegmentationDataLoader, self).__init__(
            dataset,
            self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

    def set_default(self):
        self.config.update(dict(
            image_dir='',
            mask_dir='',
            dataset_name='generic',
            num_classes=2,
            class_values=None,
            image_extension='.png',
            mask_extension='.png',
            patch_config=dict(
                patch_size=896,
                stride=512,
            ),
            transforms=[
                GenericRemoveColorMap(),
                segm.RandomHorizontalFlip(0.5),
                segm.RandomVerticalFlip(0.5),
                segm.RandomRotate90K((0, 1, 2, 3)),
                segm.FixedPad((896, 896), 255),
                segm.ToTensor(True),
                comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
            ],
            batch_size=1,
            num_workers=0,
            training=True
        ))


class GenericImageFolderDataset(Dataset):
    """Generic dataset for inference"""
    def __init__(self, image_dir, mask_dir=None, class_values=None, num_classes=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fp_list = find_images_with_extensions(image_dir)
        self.rm_color = GenericRemoveColorMap(class_values=class_values, num_classes=num_classes)

    def __getitem__(self, idx):
        image_np = imread(self.fp_list[idx])
        
        if self.mask_dir is not None:
            # Try to find mask with various extensions
            base_name = os.path.splitext(os.path.basename(self.fp_list[idx]))[0]
            mask_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp', '.gif']
            mask_np = None
            
            for ext in mask_extensions:
                mask_name = base_name + ext
                mask_fp = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_fp):
                    mask_np = imread(mask_fp)
                    _, mask = self.rm_color(None, mask_np)
                    mask_np = np.array(mask, copy=False)
                    break
                # Also try uppercase
                mask_name = base_name + ext.upper()
                mask_fp = os.path.join(self.mask_dir, mask_name)
                if os.path.exists(mask_fp):
                    mask_np = imread(mask_fp)
                    _, mask = self.rm_color(None, mask_np)
                    mask_np = np.array(mask, copy=False)
                    break
        else:
            mask_np = None
            
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=2)
            
        return image_np, mask_np, os.path.basename(self.fp_list[idx])

    def __len__(self):
        return len(self.fp_list)
