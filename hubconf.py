import torch.nn as nn
from simplecv.module import fpn
from simplecv.util import checkpoint
from simplecv.core.config import AttrDict

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

dependencies = ['torch']

from module.farseg import FarSeg

model_urls = {
    # Model URLs for generic datasets
}


def farseg_resnet50(pretrained=False, progress=True, **kwargs):
    """
    FarSeg with ResNet50 backbone for generic datasets
    """
    if pretrained:
        raise ValueError("Pretrained models not available for generic datasets. Please train your own model.")
    
    # Use generic configuration
    config = AttrDict()
    # Add your generic model configuration here
    return FarSeg(config)
