import torch
import os.path as osp
from typing import List
from torch_geometric.data import Dataset
import numpy as np
from utils.data.DataPostProcessor import DataPostProcessor

from utils.data.MyData import MyData


class LargeDataset(Dataset):
    def __init__(self, file_locator: str, config_args: dict, split: str = None, data_root=None, mdn=False, **kwargs):
        if not osp.exists(file_locator):
            file_locator = osp.join(data_root, file_locator)
            if split is not None:
                split = osp.join(data_root, split)
            assert osp.exists(file_locator)
        self.file_locator_path = file_locator
        self.file_locator = torch.load(file_locator)
        self.mdn = mdn
        self.config_args = config_args
        self.post_processor = DataPostProcessor(config_args)

        # supports both abs path and rel path stored in processed folder
        if split is None:
            self.train_index = None
            self.val_index = None
            self.valid_index = None
            self.test_index = torch.arange(len(self.file_locator))
        else:
            if not osp.exists(split):
                # for backward compatibility only
                split = osp.join(data_root, "processed", osp.basename(split))
            split = torch.load(split)
            for key in split.keys():
                assert key.endswith("_index")
                val = torch.as_tensor(split[key]).long() if split[key] is not None else None
                setattr(self, key, val)
            if self.val_index is None:
                valid_size = 1000
                np.random.seed(2333)
                perm_matrix = np.random.permutation(len(self.train_index))
                self.val_index = self.train_index[perm_matrix[-valid_size:]]
                self.train_index = self.train_index[perm_matrix[:-valid_size]]
            self.valid_index = self.val_index

        # we need to drop some properties so that the model can be trained together
        self.drop = set()
        if osp.basename(file_locator) == "PL2020-polarH_10$10$6.loc.pth":
            self.post_processor.drop_key("source")
        super().__init__(data_root, None, None, None)

    def len(self):
        return len(self.file_locator)

    def get(self, idx, process=True):
        data = torch.load(self.file_locator[idx])
        data = MyData.from_data(data)
        if process:
            data = self.post_processor(data, idx)
        return data

    @property
    def processed_file_names(self):
        return [self.file_locator_path]

    # no more stupid folder creation and downloads
    @property
    def raw_dir(self) -> str:
        return "."

    @property
    def processed_dir(self) -> str:
        return "."

    @property
    def processed_paths(self):
        return self.file_locator
