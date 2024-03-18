import logging
import os.path as osp
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from ase.units import Hartree, eV

hartree2ev = Hartree / eV


class DummyIMDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, split=None, sub_ref=False, convert_unit=True, **kwargs):
        self.sub_ref = sub_ref
        self.dataset_name = dataset_name
        self.split = split
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_index, self.val_index, self.test_index = None, None, None
        if split is not None:
            split_data = torch.load(self.processed_paths[1])
            self.test_index = torch.as_tensor(split_data["test_index"])
            rand_val_index = False
            if ("valid_index" not in split_data) and ("val_index" not in split_data):
                rand_val_index = True
            elif "val_index" in split_data and split_data["val_index"] is None:
                rand_val_index = True
            if rand_val_index:
                warnings.warn("You are randomly generating valid set from training set")
                train_index = torch.as_tensor(split_data["train_index"]).long()
                np.random.seed(2333)
                perm_matrix = np.random.permutation(len(train_index))
                self.train_index = train_index[perm_matrix[:-1000]]
                self.val_index = train_index[perm_matrix[-1000:]]
            else:
                if split_data["train_index"] is not None:
                    self.train_index = torch.as_tensor(split_data["train_index"]).long()
                else:
                    self.train_index = None

                if split_data["test_index"] is not None:
                    self.test_index = torch.as_tensor(split_data["test_index"]).long()
                else:
                    self.test_index = None

                for name in ["val_index", "valid_index"]:
                    if name in split_data.keys():
                        if split_data[name] is not None:
                            self.val_index = torch.as_tensor(split_data[name]).long()
                        else:
                            self.val_index = None
        if self.sub_ref:
            warnings.warn("sub_ref is deprecated")
            preprocess_dataset(osp.join(osp.dirname(__file__), "GaussUtils"), self, convert_unit)

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name, self.split] if self.split is not None else [self.dataset_name]

    def download(self):
        pass

    def process(self):
        pass


def subtract_ref(dataset, save_path, use_jianing_ref=True, data_root="./data", convert_unit=True):
    """
    Subtracting reference energy, the result is in eV unit
    :param convert_unit:  Convert gas from hartree to ev. Set to false if it is already in ev
    :param data_root:
    :param dataset:
    :param save_path:
    :param use_jianing_ref:
    :return:
    """
    if save_path:
        logging.info("We prefer to subtract reference on the fly rather than save the file!")
        print("We prefer to subtract reference on the fly rather than save the file!")
    if save_path is not None and osp.exists(save_path):
        raise ValueError("cannot overwrite existing file!!!")
    if use_jianing_ref:
        ref_data = np.load(osp.join(data_root, "atomref.B3LYP_631Gd.10As.npz"))
        u0_ref = ref_data["atom_ref"][:, 1]
    else:
        ref_data = pd.read_csv(osp.join(data_root, "raw/atom_ref_gas.csv"))
        u0_ref = np.zeros(96, dtype=np.float)
        for i in range(ref_data.shape[0]):
            u0_ref[int(ref_data.iloc[i]["atom_num"])] = float(ref_data.iloc[i]["energy(eV)"])
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        total_ref = u0_ref[data.Z].sum()
        for prop in ["watEnergy", "octEnergy", "gasEnergy"]:
            energy = getattr(data, prop)
            if convert_unit:
                energy *= hartree2ev
            energy -= total_ref
    if save_path is not None:
        torch.save((dataset.data, dataset.slices), save_path)


def preprocess_dataset(data_root, data_provider, convert_unit, logger=None):
    # this "if" is because of my stupid decisions of subtracting reference beforehand in the "frag9to20_all" dataset
    # but later found it better to subtract it on the fly
    for name in ["gasEnergy", "watEnergy", "octEnergy"]:
        if name in data_provider[0]:
            subtract_ref(data_provider, None, data_root=data_root, convert_unit=convert_unit)
            if logger is not None:
                logger.info("{} max: {}".format(name, getattr(data_provider.data, name).max().item()))
                logger.info("{} min: {}".format(name, getattr(data_provider.data, name).min().item()))
            break


def concat_im_datasets(root: str, datasets: List[str], out_name: str):
    data_list = []
    for dataset in datasets:
        dummy_dataset = DummyIMDataset(root, dataset)
        for i in tqdm(range(len(dummy_dataset)), dataset):
            data_list.append(dummy_dataset[i])
    print("saving... it is recommended to have 32GB memory")
    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list),
               osp.join(root, "data/processed", out_name))


if __name__ == '__main__':
    test_data = DummyIMDataset(root="data", dataset_name="freesolv_mmff_pyg.pt", split="freesolv_mmff_pyg_split.pt")
    print("")

