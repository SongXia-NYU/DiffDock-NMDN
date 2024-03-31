from glob import glob
import os.path as osp

import pandas as pd
import torch

from utils.eval.tester import Tester
from utils.utils_functions import get_device


class TestedFolderReader(Tester):
    def __init__(self, test_folder=None):
        test_folder = glob(test_folder)
        assert len(test_folder) == 1, str(test_folder)
        test_folder = test_folder[0]
        super(TestedFolderReader, self).__init__(osp.dirname(test_folder))
        self._test_dir = test_folder

        self._result_mapper = None
        self._result_file_mapper = None
        self._record_mapper = None
        self.load_chk()

    def only_record(self):
        keys = list(self.record_mapper.keys())
        assert len(keys) == 1, keys
        return self.record_mapper[keys[0]]

    @property
    def result_file_mapper(self):
        if self._result_file_mapper is None:
            mapper = {}
            for split in ["val", "test"]:
                fs = glob(osp.join(self.save_root, f"loss_*_{split}.pt"))
                if len(fs) == 0 and split == "val":
                    continue
                if len(fs) == 1:
                    mapper[split] = fs[0]
                else:
                    for loss_file in fs:
                        subset_name = osp.basename(loss_file).split("loss_")[-1].split(f"_{split}.pt")[0]
                        mapper[f"{split}@{subset_name}"] = loss_file
            self._result_file_mapper = mapper
        return self._result_file_mapper

    @property
    def result_mapper(self):
        if self._result_mapper is None:
            mapper = {}
            for split in self.result_file_mapper:
                mapper[split] = torch.load(self.result_file_mapper[split], map_location=get_device())
            self._result_mapper = mapper
        return self._result_mapper

    def record_key2result(self, key: str, split="test") -> dict:
        if len(self.record_mapper.keys()) == 1:
            return self.result_mapper[split]
        
        return self.result_mapper[f"{split}@{key}"]

    @property
    def record_mapper(self):
        if self._record_mapper is None:
            mapper = {}
            for csv in glob(osp.join(self.save_root, "record_name_*.csv")):
                df = pd.read_csv(csv)
                key = osp.basename(csv).split("record_name_")[-1].split(".csv")[0]
                mapper[key] = df
            self._record_mapper = mapper
        return self._record_mapper
