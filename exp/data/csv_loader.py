from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from .utils import MetaData


class _CSVDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, is_regression: bool):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.is_regression = is_regression
        if is_regression:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class _ObsDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class _CSVBatch:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, is_regression: bool):
        self.X = features
        obs = _ObsDict({"cell_type": labels, "target": labels.float()})
        self.obs = obs


class _CSVAnnLoaderLike:
    def __init__(self, dataset: _CSVDataset, batch_size: int, shuffle: bool, device: str, is_regression: bool):
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.device = device
        self.is_regression = is_regression

    def __iter__(self):
        for features, labels in self.loader:
            yield _CSVBatch(features.to(self.device), labels.to(self.device), self.is_regression)

    def __len__(self):
        return len(self.loader)


def generate_csv_loader(
    path: str,
    label_column: str = "label",
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = "cpu",
    random_state: int = 42,
) -> MetaData:
    df = pd.read_csv(path)
    logger.info(f"Read CSV: {df.shape[0]} rows x {df.shape[1]} columns from {path}")

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != label_column]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_column].values

    is_regression = False
    try:
        numeric_vals = pd.to_numeric(pd.Series(y_raw))
        n_unique = numeric_vals.nunique()
        if n_unique > len(y_raw) * 0.5:
            is_regression = True
    except (ValueError, TypeError):
        pass

    if is_regression:
        y = numeric_vals.values.astype(np.float32)
        label_mapping = {0: "regression_target"}
        cls_num = 1
        logger.info("Detected regression target (continuous values)")
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw).astype(np.longlong)
        label_mapping = {int(v): str(k) for k, v in zip(encoder.classes_, encoder.transform(encoder.classes_))}
        cls_num = len(label_mapping)
        logger.info(f"Detected classification target ({cls_num} classes): {label_mapping}")

    stratify = y if not is_regression and cls_num > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    train_dataset = _CSVDataset(X_train, y_train, is_regression)
    test_dataset = _CSVDataset(X_test, y_test, is_regression)
    train_loader = _CSVAnnLoaderLike(train_dataset, batch_size, shuffle, device, is_regression)
    test_loader = _CSVAnnLoaderLike(test_dataset, batch_size, shuffle=False, device=device, is_regression=is_regression)

    return MetaData(
        name=os.path.splitext(os.path.basename(path))[0],
        feature_dim=X.shape[1],
        train_loader=train_loader,
        test_loader=test_loader,
        label_mapping=label_mapping,
        cls_num=cls_num,
        random_state=random_state,
        shuffle=shuffle,
        batch_size=batch_size,
    )
