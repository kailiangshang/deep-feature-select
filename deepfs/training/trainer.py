"""
Trainer for feature selection models.
"""
from __future__ import annotations

from typing import Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from deepfs.core import BaseSelector, SparsityLoss


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    sparsity_weight: float = 0.1
    device: str = "cpu"
    verbose: bool = True
    print_every: int = 10


class FeatureSelectionTrainer:
    """
    Trainer for feature selection models.
    
    Handles the training loop for models with feature selectors,
    including sparsity regularization and temperature scheduling.
    
    Parameters
    ----------
    model : nn.Module
        Model containing a feature selector
    selector : BaseSelector
        Feature selector module
    config : TrainConfig
        Training configuration
        
    Examples
    --------
    >>> model = MyModel(selector)
    >>> trainer = FeatureSelectionTrainer(model, selector, config)
    >>> trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        selector: BaseSelector,
        config: Optional[TrainConfig] = None
    ):
        self.model = model
        self.selector = selector
        self.config = config or TrainConfig()
        
        self.device = self.config.device
        self.model.to(self.device)
        self.selector.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        self.callbacks: List[Any] = []
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "sparsity_loss": []
        }
    
    def add_callback(self, callback: Any) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def _call_callbacks(self, method: str, *args, **kwargs) -> None:
        """Call method on all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method):
                getattr(callback, method)(*args, **kwargs)
    
    def _get_sparsity_loss(self) -> torch.Tensor:
        """Get sparsity loss from selector."""
        if hasattr(self.selector, 'sparsity_loss'):
            loss = self.selector.sparsity_loss()
            if isinstance(loss, SparsityLoss):
                return loss.total
            return loss
        return torch.tensor(0.0, device=self.device)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            task_loss = criterion(outputs, batch_y)
            
            # Sparsity loss
            sparsity_loss = self._get_sparsity_loss()
            
            # Combined loss
            loss = task_loss + self.config.sparsity_weight * sparsity_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "task_loss": total_task_loss / num_batches,
            "sparsity_loss": total_sparsity_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        criterion: Callable
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        criterion : Callable, optional
            Loss function (default: MSELoss for regression, CrossEntropyLoss for classification)
            
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        # Callback: on_train_begin
        self._call_callbacks("on_train_begin", trainer=self)
        
        for epoch in range(self.config.epochs):
            # Callback: on_epoch_begin
            self._call_callbacks("on_epoch_begin", trainer=self, epoch=epoch)
            
            # Train
            train_metrics = self.train_epoch(train_loader, criterion)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, criterion)
            else:
                val_metrics = {}
            
            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["sparsity_loss"].append(train_metrics["sparsity_loss"])
            if val_metrics:
                self.history["val_loss"].append(val_metrics["val_loss"])
            
            # Callback: on_epoch_end
            self._call_callbacks(
                "on_epoch_end", 
                trainer=self, 
                epoch=epoch,
                metrics={**train_metrics, **val_metrics}
            )
            
            # Print progress
            if self.config.verbose and (epoch + 1) % self.config.print_every == 0:
                msg = f"Epoch {epoch+1}/{self.config.epochs}"
                msg += f" - Loss: {train_metrics['loss']:.4f}"
                if val_metrics:
                    msg += f" - Val Loss: {val_metrics['val_loss']:.4f}"
                msg += f" - Sparsity: {train_metrics['sparsity_loss']:.4f}"
                print(msg)
        
        # Callback: on_train_end
        self._call_callbacks("on_train_end", trainer=self)
        
        return self.history
    
    def get_selected_features(self) -> np.ndarray:
        """Get selected feature indices."""
        self.model.eval()
        result = self.selector.get_selection_result()
        return result.selected_indices[result.selected_indices >= 0]