from __future__ import annotations

import os
import random

import numpy as np
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from anndata.experimental.multi_files import AnnCollection
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utils import MetaData

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def generate_train_test_loader(
    name: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = "cpu",
    random_state: int = 42,
    feature_indexs: list[int] | None = None,
) -> MetaData:
    path = os.path.join(DATA_DIR, f"{name}.h5ad")
    adata = sc.read_h5ad(path)
    logger.info(f"[{name}] {adata.shape[0]} cells x {adata.shape[1]} genes from {path}")

    encoder_ct = LabelEncoder()
    encoder_ct.fit(adata.obs["cell_type"])
    adata.obs["cell_type"] = encoder_ct.transform(adata.obs["cell_type"]).astype(
        np.longlong
    )
    label_mapping = {
        v: k
        for k, v in zip(encoder_ct.classes_, encoder_ct.transform(encoder_ct.classes_))
    }
    logger.info(f"[{name}] {len(label_mapping)} classes")

    if feature_indexs is not None:
        adata = adata[:, feature_indexs]
        logger.info(f"[{name}] subset to {len(feature_indexs)} features")

    dataset = AnnCollection([adata])
    train_ad, test_ad = train_test_split(
        dataset,
        random_state=random_state,
        test_size=test_size,
        stratify=adata.obs["cell_type"],
    )

    train_loader = AnnLoader(
        train_ad, batch_size=batch_size, shuffle=shuffle, use_cuda=(device == "cuda")
    )
    test_loader = AnnLoader(
        test_ad, batch_size=batch_size, shuffle=shuffle, use_cuda=(device == "cuda")
    )

    return MetaData(
        name=name,
        train_loader=train_loader,
        test_loader=test_loader,
        label_mapping=label_mapping,
        cls_num=len(label_mapping),
        random_state=random_state,
        shuffle=shuffle,
        batch_size=batch_size,
        feature_dim=adata.shape[1],
    )
