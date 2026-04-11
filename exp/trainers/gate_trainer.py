from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report

from deepfs.core.base import GateFeatureModule
from exp.utils import seed_all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class GateTrainer:
    def __init__(
        self,
        model: GateFeatureModule,
        classifier: nn.Module,
        sparse_loss_weight: float = 1.0,
        lr: float = 1e-4,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.model = model.to(device)
        self.classifier = classifier.to(device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.classifier.parameters()), lr=lr
        )
        self.sparse_loss_weight = sparse_loss_weight
        self.seed = seed
        self.device = device

    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        self.classifier.train()
        total_cls_loss = 0.0
        total_sparse_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            data, target = batch.X, batch.obs["cell_type"]
            self.optimizer.zero_grad()
            features = self.model(data)
            output = self.classifier(features)
            loss_cls = F.cross_entropy(output, target, reduction="mean")
            sparsity = self.model.sparsity_loss()
            loss = loss_cls + self.sparse_loss_weight * sparsity.total
            loss.backward()
            self.optimizer.step()
            total_cls_loss += loss_cls.item()
            total_sparse_loss += sparsity.total.item()
            num_batches += 1
        self.model.update_temperature(epoch)
        return total_cls_loss / max(num_batches, 1), total_sparse_loss / max(num_batches, 1)

    def _evaluate(self, test_loader):
        self.model.eval()
        self.classifier.eval()
        if test_loader is None:
            return 0.0, {}
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in test_loader:
                data, target = batch.X, batch.obs["cell_type"]
                features = self.model(data)
                output = self.classifier(features)
                preds = output.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        report = classification_report(all_targets, all_preds, output_dict=True)
        return report["accuracy"], report

    def fit(self, train_loader, epochs, test_loader=None):
        seed_all(self.seed)
        records = []
        for epoch in range(epochs):
            loss_cls, loss_sparse = self._train_epoch(train_loader, epoch)
            acc, report = self._evaluate(test_loader)
            result = self.model.get_selection_result()
            records.append(
                {
                    "epoch": epoch + 1,
                    "loss_cls": loss_cls,
                    "loss_sparsity": loss_sparse,
                    "accuracy": acc,
                    "num_selected": result.num_selected,
                }
            )
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Loss: {loss_cls:.4f}, "
                f"Sparse: {loss_sparse:.4f}, "
                f"Accuracy: {acc:.4f}, "
                f"Features: {result.num_selected}"
            )
        return pd.DataFrame(records)
