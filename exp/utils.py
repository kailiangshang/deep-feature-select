from __future__ import annotations

from typing import NamedTuple

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


class TorchLossDict(NamedTuple):
    """
    A named tuple representing the loss dictionary.

    Attributes:
        cls_loss_name (str): The name of the classification loss.
        sparcity_loss_name (list[str]): The names of the sparcity losses.
        cls_loss_tensors (torch.Tensor): The classification loss tensors.
        sparcity_loss_tensors (list[torch.Tensor]): The sparcity loss tensors.
    """
    cls_loss_name: str
    sparcity_loss_name: list[str]
    cls_loss_tensors: torch.Tensor
    sparcity_loss_tensors: list[torch.Tensor]
    

def seed_all(seed: int) -> None:
    """
    Set the seed for all random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    return None


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device: str = 'cpu'):
        super(MLPClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 添加 dropout 层以防止过拟合
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def cross_entropy_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, reduction='mean')
    

class Result:
    
    def __init__(self):
        
        self._feature_num = []
        self._accuracy = []
        self._macro_precision = []
        self._macro_recall = []
        self._macro_f1 = []
        self._weighted_precision = []
        self._weighted_recall = []
        self._weighted_f1 = []
    
    @property
    def feature_num(self):
        return self._feature_num
    
    @feature_num.setter
    def feature_num(self, value):
        self._feature_num.append(value)
        
    @property
    def accuracy(self):
        return self._accuracy
    
    @accuracy.setter
    def accuracy(self, value):
        self._accuracy.append(value)
        
    @property
    def macro_precision(self):
        return self._macro_precision
    
    @macro_precision.setter
    def macro_precision(self, value):
        self._macro_precision.append(value)
        
    @property
    def macro_recall(self):
        return self._macro_recall
    
    @macro_recall.setter
    def macro_recall(self, value):
        self._macro_recall.append(value)
        
    @property
    def macro_f1(self):
        return self._macro_f1
    
    @macro_f1.setter
    def macro_f1(self, value):
        self._macro_f1.append(value)
        
    @property
    def weighted_precision(self):
        return self._weighted_precision
    
    @weighted_precision.setter
    def weighted_precision(self, value):
        self._weighted_precision.append(value)
        
    @property
    def weighted_recall(self):
        return self._weighted_recall
    
    @weighted_recall.setter
    def weighted_recall(self, value):
        self._weighted_recall.append(value)
        
    @property
    def weighted_f1(self):
        return self._weighted_f1
    
    @weighted_f1.setter
    def weighted_f1(self, value):
        self._weighted_f1.append(value)    
    
    def to_df(self):
        return pd.DataFrame({
            'feature_num': self.feature_num,
            'macro_precision': self.macro_precision,
            'macro_recall': self.macro_recall,
            'macro_f1': self.macro_f1,
            'weighted_precision': self.weighted_precision,
            'weighted_recall': self.weighted_recall,
            'weighted_f1': self.weighted_f1,
            'accuracy': self.accuracy,
        })
    