import argparse
import copy
import glob
import os
import os.path as osp
import time
import traceback
from typing import List

import pandas as pd
import torch
import torch_geometric.data
from torch_geometric.data import Data, HeteroData
import numpy as np

from geometry_processors.lazy_property import lazy_property

class PyGCollateProcessor:
    def __init__(self, pygs: List[str] = None, save_pyg: str = None, data_list: List[Data] = None) -> None:
        self.pygs = pygs
        self.save_pyg = save_pyg
        self.data_list_raw = data_list
        self.save_split: str = None
        self.extend_load: bool = False
        self.ref_csv: str = None
        self.all_train: bool = False
        self.all_test: bool = False
        self.n_max_pad: bool = None

    @lazy_property
    def data_list(self) -> List[Data]:
        if self.data_list_raw is not None:
            return self.data_list_raw
        
        data_list = []
        assert self.pygs is not None

        if self.ref_csv is not None:
            ref_df = pd.read_csv(self.ref_csv, dtype={"sample_id": str, "FileHandle": str}) \
                .rename({"FileHandle": "sample_id"}, axis=1).set_index("sample_id")
        else:
            ref_df = None

        for pyg in self.pygs:
            try:
                d = torch.load(pyg)
            except Exception as e:
                print(f"Error loading {pyg}: {e}")
                print(traceback.format_exc())
                continue

            if ref_df is not None:
                sample_id = str(osp.basename(pyg).split(".")[0])
                this_info = ref_df.loc[[sample_id], :]
                for key in this_info.columns:
                    if self.extend_load:
                        for _d in d:
                            setattr(_d, key, this_info[key].item())
                    else:
                        setattr(d, key, this_info[key].item())
            
            if not hasattr(d, "file_handle"):
                d.file_handle = ".".join(osp.basename(pyg).split(".")[:-1])

            if self.extend_load:
                data_list.extend(d)
            else:
                data_list.append(d)
        return data_list
    
    def add_padding(self):
        if self.n_max_pad is None:
            return
        
        def padding_homo_data(data: Data):
            n_second_dim = data.R_prot_pad.shape[1]
            if n_second_dim >= self.n_max_pad:
                return
            R_prot_pad = data.R_prot_pad.numpy()
            R_prot_pad = np.pad(R_prot_pad, ((0, 0), (0, self.n_max_pad-n_second_dim), (0, 0)), 
                                constant_values=np.nan)
            data.R_prot_pad = torch.as_tensor(R_prot_pad)
            Z_prot_pad = data.Z_prot_pad.numpy()
            Z_prot_pad = np.pad(Z_prot_pad, ((0, 0), (0, self.n_max_pad - n_second_dim)), constant_values=-1)
            data.Z_prot_pad = torch.as_tensor(Z_prot_pad)

        def padding_hetero_data(data: HeteroData):
            n_second_dim = data["protein"].R.shape[1]
            if n_second_dim >= self.n_max_pad:
                return
            R_prot_pad = data["protein"].R.numpy()
            R_prot_pad = np.pad(R_prot_pad, ((0, 0), (0, self.n_max_pad-n_second_dim), (0, 0)), 
                                constant_values=np.nan)
            data["protein"].R = torch.as_tensor(R_prot_pad)
            Z_prot_pad = data["protein"].Z.numpy()
            Z_prot_pad = np.pad(Z_prot_pad, ((0, 0), (0, self.n_max_pad - n_second_dim)), constant_values=-1)
            data["protein"].Z = torch.as_tensor(Z_prot_pad)
            
        for d in self.data_list:
            if isinstance(d, HeteroData):
                padding_hetero_data(d)
            else:
                padding_homo_data(d)

    def try_save_split(self):
        train_index = []
        test_index = []

        if self.save_split is None:
            return
        if self.all_train:
            split = {
                "train_index": torch.arange(len(self.data_list)),
                "val_index": None,
                "test_index": None
            }
        elif self.all_test:
            split = {
                "train_index": None,
                "val_index": None,
                "test_index": torch.arange(len(self.data_list))
            }
        else:
            for num, pyg in enumerate(self.data_list):
                if pyg.split == "train":
                    train_index.append(num)
                else:
                    test_index.append(num)
            split = {
                "train_index": torch.as_tensor(train_index),
                "val_index": None,
                "test_index": torch.as_tensor(test_index)
            }
        torch.save(split, self.save_split)

    def run(self):
        t0 = time.time()
        self.add_padding()

        print(f"Total time loading data into memory: {time.time() - t0}]")
        t0 = time.time()

        os.makedirs(osp.dirname(self.save_pyg), exist_ok=True)
        self.try_save_split()

        print(f"Total time saving split: {time.time() - t0}]")
        t0 = time.time()

        data_concat = torch_geometric.data.InMemoryDataset.collate(self.data_list)
        print(data_concat)
        torch.save(data_concat, self.save_pyg)

        print(f"Total time collate: {time.time() - t0}]")
        t0 = time.time()


def concat_pyg(pygs: list = None, save_pyg: str = None, data_list=None, save_split=None, extend_load=False,
               ref_csv=None, all_train=False, all_test=False, n_max_pad=None, **kwargs):
    processor = PyGCollateProcessor(pygs, save_pyg, data_list)
    processor.save_split = save_split
    processor.extend_load = extend_load
    processor.ref_csv = ref_csv
    processor.all_train = all_train
    processor.all_test = all_test
    processor.n_max_pad = n_max_pad

    processor.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygs", default=None)
    parser.add_argument("--pyg_folder", default=None)
    parser.add_argument("--save_pyg")
    parser.add_argument("--save_split", default=None)
    parser.add_argument("--extend_load", action="store_true")
    parser.add_argument("--ref_csv", default=None)
    parser.add_argument("--all_train", action="store_true")
    parser.add_argument("--all_test", action="store_true")

    args = parser.parse_args()
    args = vars(args)
    processed_args = copy.deepcopy(args)

    if args["pygs"] is not None:
        num = None
        for name in ["pygs"]:
            if args[name] is None:
                processed_args[name] = [None] * num
                continue

            with open(args[name]) as f:
                processed_args[name] = f.read().split()

            if num is None:
                num = len(processed_args[name])

    if args["pyg_folder"] is not None:
        pygs = glob.glob(osp.join(args["pyg_folder"], "*.pyg"))
        pygs.sort()
        processed_args["pygs"] = pygs

    concat_pyg(**processed_args)


if __name__ == '__main__':
    main()
