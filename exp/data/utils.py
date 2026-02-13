from dataclasses import dataclass
from anndata.experimental.pytorch import AnnLoader

import numpy as np
import random
from torch.utils.data import Dataset, DataLoader



@dataclass
class MetaData:
    name: str
    feature_dim: int
    train_loader: AnnLoader
    test_loader: AnnLoader
    label_mapping: dict
    cls_num: int
    random_state: int
    shuffle: bool
    batch_size: int
    
    def __post_init__(self):
        self._val_loader = None
    
    def generate_val_loader(self):
        """
        从训练集中抽取 20% 的数据作为验证集，并返回一个 PyTorch DataLoader。
        
        Parameters:
        ----------
        batch_size : int, optional
            验证集的批量大小，默认为 32。
        shuffle : bool, optional
            是否打乱验证集数据，默认为 True。
        
        Returns:
        -------
        DataLoader
            验证集的 DataLoader。
        """
        
        class CustomDataset(Dataset):
            
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            
            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        random.seed(self.random_state)
        val_data = []
        val_labels = []

        for batch_idx, batch in enumerate(self.train_loader):
            # 随机抽取 20% 的数据
            batch_size_total = batch.X.size(0)
            sample_indices = random.sample(range(batch_size_total), int(batch_size_total * 0.2))
            sampled_X = batch.X[sample_indices]
            sampled_y = batch.obs['cell_type'][sample_indices]

            # 将抽取的数据和标签存储到列表中
            val_data.append(sampled_X.cpu().numpy())
            val_labels.extend(sampled_y.cpu().tolist())

        # 合并所有批次的数据和标签
        val_data = np.vstack(val_data)
        val_labels = np.array(val_labels)

        # 创建自定义 Dataset
        val_dataset = CustomDataset(val_data, val_labels)

        # 创建 DataLoader
        self._val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        return self
    
    @property
    def val_loader(self):
        if self._val_loader is None:
            self.generate_val_loader()
        
        return self._val_loader