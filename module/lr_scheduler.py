"""
Custom Learning Rate Schedulers for FarSeg/FarSeg++ Training

Implements various LR scheduling strategies compatible with validation-based training.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    
    Decays learning rate from base_lr to end_lr using polynomial function:
        lr = base_lr * (1 - iter/max_iters)^power
    
    Compatible with early stopping - can be used even if training stops before max_iters.
    """
    
    def __init__(self, optimizer, max_iters, power=0.9, end_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            max_iters (int): Maximum number of iterations
            power (float): Polynomial power. Default: 0.9
            end_lr (float): Minimum learning rate. Default: 0.0
            last_epoch (int): The index of last epoch. Default: -1
        """
        self.max_iters = max_iters
        self.power = power
        self.end_lr = end_lr
        super(PolynomialLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        
        factor = (1 - self.last_epoch / self.max_iters) ** self.power
        return [base_lr * factor + self.end_lr * (1 - factor) 
                for base_lr in self.base_lrs]


class PolynomialWarmupLRScheduler(_LRScheduler):
    """
    Polynomial LR with warmup period.
    
    Linearly increases LR during warmup, then applies polynomial decay.
    """
    
    def __init__(self, optimizer, max_iters, warmup_iters=1000, power=0.9, 
                 end_lr=0.0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            max_iters (int): Maximum number of iterations
            warmup_iters (int): Number of warmup iterations. Default: 1000
            power (float): Polynomial power. Default: 0.9
            end_lr (float): Minimum learning rate. Default: 0.0
            last_epoch (int): The index of last epoch. Default: -1
        """
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power
        self.end_lr = end_lr
        super(PolynomialWarmupLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            factor = self.last_epoch / self.warmup_iters
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = min(progress, 1.0)
            factor = (1 - progress) ** self.power
            return [base_lr * factor + self.end_lr * (1 - factor) 
                    for base_lr in self.base_lrs]


class CosineAnnealingWarmupLRScheduler(_LRScheduler):
    """
    Cosine annealing with warmup and optional restarts.
    """
    
    def __init__(self, optimizer, max_iters, warmup_iters=1000, 
                 eta_min=1e-6, restart_iters=None, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer
            max_iters (int): Maximum number of iterations
            warmup_iters (int): Number of warmup iterations. Default: 1000
            eta_min (float): Minimum learning rate. Default: 1e-6
            restart_iters (int): Period for cosine restart. If None, no restart. Default: None
            last_epoch (int): The index of last epoch. Default: -1
        """
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.eta_min = eta_min
        self.restart_iters = restart_iters
        super(CosineAnnealingWarmupLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            factor = self.last_epoch / self.warmup_iters
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            if self.restart_iters:
                # With restarts
                progress = (self.last_epoch - self.warmup_iters) % self.restart_iters
                T_cur = progress
                T_max = self.restart_iters
            else:
                # Without restarts
                T_cur = self.last_epoch - self.warmup_iters
                T_max = self.max_iters - self.warmup_iters
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * T_cur / T_max)) / 2
                for base_lr in self.base_lrs
            ]


def build_lr_scheduler(optimizer, config, total_iters=None):
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config (dict): Scheduler configuration
        total_iters (int): Total training iterations (for iteration-based schedulers)
    
    Returns:
        LR scheduler instance
    
    Example config formats:
    
    1. Polynomial (iteration-based):
        {
            'type': 'poly',
            'params': {
                'power': 0.9,
                'end_lr': 0.0
            }
        }
    
    2. Polynomial with warmup:
        {
            'type': 'poly_warmup',
            'params': {
                'warmup_iters': 1000,
                'power': 0.9,
                'end_lr': 0.0
            }
        }
    
    3. ReduceLROnPlateau (validation-based):
        {
            'type': 'plateau',
            'params': {
                'mode': 'max',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            }
        }
    
    4. Cosine annealing with warmup:
        {
            'type': 'cosine_warmup',
            'params': {
                'warmup_iters': 1000,
                'eta_min': 1e-6,
                'restart_iters': 5000
            }
        }
    """
    scheduler_type = config.get('type', 'poly')
    params = config.get('params', {})
    
    if scheduler_type == 'poly':
        if total_iters is None:
            raise ValueError("total_iters must be specified for polynomial scheduler")
        return PolynomialLRScheduler(
            optimizer,
            max_iters=total_iters,
            power=params.get('power', 0.9),
            end_lr=params.get('end_lr', 0.0)
        )
    
    elif scheduler_type == 'poly_warmup':
        if total_iters is None:
            raise ValueError("total_iters must be specified for polynomial warmup scheduler")
        return PolynomialWarmupLRScheduler(
            optimizer,
            max_iters=total_iters,
            warmup_iters=params.get('warmup_iters', 1000),
            power=params.get('power', 0.9),
            end_lr=params.get('end_lr', 0.0)
        )
    
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=params.get('mode', 'max'),
            factor=params.get('factor', 0.5),
            patience=params.get('patience', 5),
            threshold=params.get('threshold', 0.001),
            min_lr=params.get('min_lr', 1e-6),
            verbose=params.get('verbose', True)
        )
    
    elif scheduler_type == 'cosine_warmup':
        if total_iters is None:
            raise ValueError("total_iters must be specified for cosine warmup scheduler")
        return CosineAnnealingWarmupLRScheduler(
            optimizer,
            max_iters=total_iters,
            warmup_iters=params.get('warmup_iters', 1000),
            eta_min=params.get('eta_min', 1e-6),
            restart_iters=params.get('restart_iters', None)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_current_lr(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']
