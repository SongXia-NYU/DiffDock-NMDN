from typing import List, Optional, Union
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader.dataloader import Collater
from torchdrug.data.dataloader import graph_collate
from torchdrug.data.protein import Protein

class PLDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=PLCollater(follow_batch, exclude_keys),
            **kwargs,
        )

class PLCollater(Collater):
    def __init__(self, follow_batch, exclude_keys):
        super().__init__(follow_batch, exclude_keys)
    
    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, Protein):
            return graph_collate(batch)
        return super().__call__(batch)
