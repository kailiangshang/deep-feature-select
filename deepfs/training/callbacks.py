"""
Callbacks for feature selection training.
"""
from __future__ import annotations

from typing import Any, Optional
import os

import torch
import numpy as np


class TemperatureCallback:
    """
    Callback to update temperature during training.
    
    Automatically calls update_temperature(epoch) on the selector
    at the beginning of each epoch.
    
    Parameters
    ----------
    selector : Any
        Feature selector with update_temperature method
    """
    
    def __init__(self, selector: Any):
        self.selector = selector
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs) -> None:
        """Update temperature at epoch start."""
        if hasattr(self.selector, 'update_temperature'):
            self.selector.update_temperature(epoch)


class LoggingCallback:
    """
    Callback to log training progress.
    
    Parameters
    ----------
    log_dir : str, optional
        Directory to save logs
    log_every : int, default=10
        Log every N epochs
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_every: int = 10
    ):
        self.log_dir = log_dir
        self.log_every = log_every
        self.logs = []
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def on_epoch_end(
        self, 
        trainer: Any, 
        epoch: int, 
        metrics: dict, 
        **kwargs
    ) -> None:
        """Log metrics at epoch end."""
        if (epoch + 1) % self.log_every == 0:
            log_entry = {"epoch": epoch + 1, **metrics}
            self.logs.append(log_entry)
    
    def on_train_end(self, trainer: Any, **kwargs) -> None:
        """Save logs at training end."""
        if self.log_dir:
            import json
            log_path = os.path.join(self.log_dir, "training_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.logs, f, indent=2)


class EarlyStoppingCallback:
    """
    Callback for early stopping based on validation loss.
    
    Parameters
    ----------
    patience : int, default=10
        Number of epochs to wait before stopping
    min_delta : float, default=0.0
        Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(
        self, 
        trainer: Any, 
        epoch: int, 
        metrics: dict, 
        **kwargs
    ) -> None:
        """Check for early stopping."""
        val_loss = metrics.get('val_loss', None)
        
        if val_loss is None:
            return
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"Early stopping at epoch {epoch + 1}")


class ModelCheckpointCallback:
    """
    Callback to save model checkpoints.
    
    Parameters
    ----------
    save_dir : str
        Directory to save checkpoints
    save_best_only : bool, default=True
        Only save when validation loss improves
    """
    
    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True
    ):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_loss = float('inf')
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def on_epoch_end(
        self, 
        trainer: Any, 
        epoch: int, 
        metrics: dict, 
        **kwargs
    ) -> None:
        """Save checkpoint if validation loss improved."""
        val_loss = metrics.get('val_loss', None)
        
        if val_loss is None and self.save_best_only:
            return
        
        if not self.save_best_only or val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics
            }
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, path)