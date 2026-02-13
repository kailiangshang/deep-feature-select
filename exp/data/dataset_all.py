from __future__ import annotations
from dataclasses import dataclass
import random
import torch
import scanpy as sc
import anndata
from anndata.experimental.pytorch import AnnLoader
from anndata.experimental.multi_files import AnnCollection

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from loguru import logger
import numpy as np

from .utils import MetaData


def generate_train_test_loader(
    name: str,
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle: bool = True,
    device: str = "cuda",
    random_state=42,
    feature_indexs: list[int] = None,
    feature_weights: list[float] = None,
    **kwargs,
) -> MetaData:
    """
    Generate train and test loaders from a h5ad file.

    Parameters
    ----------
    path : str
        Path to the h5ad file.
    test_size : float, optional
        Size of the test set, by default 0.2.
    batch_size : int, optional
        Batch size, by default 32.
    shuffle : bool, optional
        Whether to shuffle the data, by default True.
        
    Returns
    -------
    tuple[AnnLoader, AnnLoader]
        Train and test loaders.
    """
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    path = os.path.join(current_dir, f"{name}.h5ad")
    
    pancreas_anndata = sc.read_h5ad(path)
    logger.info(f"Read {pancreas_anndata.shape[0]} cells and {pancreas_anndata.shape[1]} genes from {path}")
    logger.info(f"Label class counts: {pancreas_anndata.obs['cell_type'].value_counts()/pancreas_anndata.shape[0]}")
    
    # 将 cell_type 列进行分类编码
    encoder_cell_type = LabelEncoder()
    encoder_cell_type.fit(pancreas_anndata.obs['cell_type'])
    pancreas_anndata.obs['cell_type'] = encoder_cell_type.transform(pancreas_anndata.obs['cell_type']).astype(np.longlong)
    label_mapping = {v: k for k, v in zip(encoder_cell_type.classes_, encoder_cell_type.transform(encoder_cell_type.classes_))}
    logger.info(f"Label class mapping {label_mapping}")
    
    if feature_indexs is not None:
        pancreas_anndata = pancreas_anndata[:, feature_indexs]
        logger.info(f"Select {len(feature_indexs)} features from {pancreas_anndata.shape[1]} features")
    
    if feature_weights is not None:
        if len(feature_weights) != pancreas_anndata.shape[1]:
            raise ValueError(f"Feature weights length {len(feature_weights)} does not match number of features {pancreas_anndata.shape[1]}")
        # 将特征权重应用到数据上
        pancreas_anndata = pancreas_anndata * feature_weights
    
    # 将原始数据转换为 AnnCollection 也就是 AnnData 的集合
    pancreas_dataset = AnnCollection([pancreas_anndata])
    
    train_pancreas, test_pancreas = train_test_split(
        pancreas_dataset, random_state=random_state, test_size=test_size, stratify=pancreas_anndata.obs['cell_type']
    )

    train_loader = AnnLoader(
        train_pancreas, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        use_cuda=device == "cuda"
        )
    test_loader = AnnLoader(
        test_pancreas, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        use_cuda=device == "cuda"
        )
    pancreas_metadata = MetaData(
        name=name,
        train_loader=train_loader, 
        test_loader=test_loader, 
        label_mapping=label_mapping, 
        cls_num=len(label_mapping), 
        random_state=random_state,
        shuffle=shuffle,
        batch_size=batch_size,
        feature_dim=pancreas_anndata.shape[1],
        )
    return pancreas_metadata
    
    
if __name__ == "__main__":
    
    pancreas_metadata = generate_train_test_loader(
        "E:/桌面/DFS/test_with_mnist/feature/high-dim-fs/data/pancreas.h5ad",
        feature_indexs=[0, 1, 2]
    )

    # 生成验证集的 DataLoader
    

    # 打印验证集的 DataLoader
    for sample in pancreas_metadata.val_loader:
        print(sample)
    
