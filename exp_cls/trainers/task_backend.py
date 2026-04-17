from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, r2_score, mean_squared_error


class TaskBackend(ABC):
    @abstractmethod
    def compute_loss(self, output, target): ...

    @abstractmethod
    def get_target(self, batch): ...

    @abstractmethod
    def evaluate(self, predictions, targets) -> dict: ...

    @abstractmethod
    def predict(self, output) -> np.ndarray: ...

    @property
    @abstractmethod
    def metric_name(self) -> str: ...


class ClassificationBackend(TaskBackend):
    def compute_loss(self, output, target):
        return F.cross_entropy(output, target.long(), reduction="mean")

    def get_target(self, batch):
        return batch.obs["cell_type"]

    def evaluate(self, predictions, targets):
        report = classification_report(
            targets, predictions, output_dict=True, zero_division=0
        )
        return {"metric": report["accuracy"], "report": report}

    def predict(self, output):
        return output.argmax(dim=1).cpu().numpy()

    @property
    def metric_name(self):
        return "accuracy"


class RegressionBackend(TaskBackend):
    def compute_loss(self, output, target):
        if target.dim() < output.dim():
            target = target.unsqueeze(-1)
        return F.mse_loss(output, target.float(), reduction="mean")

    def get_target(self, batch):
        if hasattr(batch, "obs"):
            return batch.obs["target"].float()
        return batch[1].float()

    def evaluate(self, predictions, targets):
        r2 = r2_score(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        return {"metric": r2, "rmse": rmse}

    def predict(self, output):
        return output.detach().cpu().numpy()

    @property
    def metric_name(self):
        return "r2"


class ReconstructionBackend(TaskBackend):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def compute_loss(self, output, target):
        return F.mse_loss(output, target.float(), reduction="mean")

    def get_target(self, batch):
        if hasattr(batch, "X"):
            return batch.X
        return batch[0]

    def evaluate(self, predictions, targets):
        mse = mean_squared_error(targets, predictions)
        return {"metric": -mse, "mse": mse}

    def predict(self, output):
        return output.detach().cpu().numpy()

    @property
    def metric_name(self):
        return "neg_mse"


def get_task_backend(task: str, **kwargs) -> TaskBackend:
    if task == "classification":
        return ClassificationBackend()
    elif task == "regression":
        return RegressionBackend()
    elif task == "reconstruction":
        if "input_dim" not in kwargs:
            raise ValueError("ReconstructionBackend requires input_dim")
        return ReconstructionBackend(kwargs["input_dim"])
    else:
        raise ValueError(f"Unknown task: {task}")
