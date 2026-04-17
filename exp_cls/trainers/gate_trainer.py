from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deepfs.core.base import GateFeatureModule
from exp_cls.utils import seed_all
from .task_backend import TaskBackend, get_task_backend

warnings.filterwarnings("ignore")


class GateTrainer:
    def __init__(
        self,
        model: GateFeatureModule,
        head: nn.Module,
        task: str | TaskBackend = "classification",
        sparse_loss_weight: float = 1.0,
        lr: float = 1e-4,
        device: str = "cpu",
        seed: int = 0,
        **backend_kwargs,
    ):
        self.model = model.to(device)
        self.head = head.to(device)
        self.task = (
            task
            if isinstance(task, TaskBackend)
            else get_task_backend(task, **backend_kwargs)
        )
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.head.parameters()), lr=lr
        )
        self.sparse_loss_weight = sparse_loss_weight
        self.seed = seed
        self.device = device

    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        self.head.train()
        total_task_loss = 0.0
        total_sparse_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            data = (batch.X if hasattr(batch, "X") else batch[0]).to(self.device)
            target = self.task.get_target(batch).to(self.device)
            self.optimizer.zero_grad()
            features = self.model(data)
            output = self.head(features)
            task_loss = self.task.compute_loss(output, target)
            sparsity = self.model.sparsity_loss()
            loss = task_loss + self.sparse_loss_weight * sparsity.total
            loss.backward()
            self.optimizer.step()
            total_task_loss += task_loss.item()
            total_sparse_loss += sparsity.total.item()
            num_batches += 1
        self.model.update_temperature(epoch)
        return total_task_loss / max(num_batches, 1), total_sparse_loss / max(
            num_batches, 1
        )

    def _evaluate(self, test_loader):
        self.model.eval()
        self.head.eval()
        if test_loader is None:
            return 0.0, {}
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                data = (batch.X if hasattr(batch, "X") else batch[0]).to(self.device)
                target = self.task.get_target(batch).to(self.device)
                features = self.model(data)
                output = self.head(features)
                all_preds.append(self.task.predict(output))
                all_targets.append(target.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        result = self.task.evaluate(all_preds, all_targets)
        return result["metric"], result

    def fit(self, train_loader, epochs, test_loader=None):
        seed_all(self.seed)
        records = []
        for epoch in range(epochs):
            loss_task, loss_sparse = self._train_epoch(train_loader, epoch)
            metric, _ = self._evaluate(test_loader)
            sel_result = self.model.get_selection_result()
            records.append(
                {
                    "epoch": epoch + 1,
                    "loss_task": loss_task,
                    "loss_sparsity": loss_sparse,
                    self.task.metric_name: metric,
                    "num_selected": sel_result.num_selected,
                }
            )
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {loss_task:.4f}, "
                f"Sparse: {loss_sparse:.4f}, "
                f"{self.task.metric_name}: {metric:.4f}, "
                f"Features: {sel_result.num_selected}"
            )
        return pd.DataFrame(records)
