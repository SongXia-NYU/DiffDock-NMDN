from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import numpy as np
import pandas as pd
import torch
from glob import glob

from utils.data.DummyIMDataset import DummyIMDataset, VSDummyIMDataset


class DeltaLearningDataset(DummyIMDataset):
    """
    An abstract class implementing basic functionality of delta machine learning
    """
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, **kwargs)

        self._fl2base_mapper = None
        self._idx2base_mapper = None
        self._excluded_idx = None

        self.remove_excluded_idx()

    def get(self, idx: int, process=True) -> Data:
        d = super().get(idx, process)
        d.base_value = self.idx2base_mapper[idx]
        return d

    def remove_excluded_idx(self):
        # remove the indices without a lin_f9 score
        for name in ["train_index", "val_index", "test_index"]:
            this_idx = getattr(self, name)
            if this_idx is None:
                continue

            this_idx = set(this_idx.numpy().tolist())
            this_idx = this_idx.difference(self.excluded_idx)
            this_idx = torch.as_tensor(list(this_idx))
            setattr(self, name, this_idx)

    @property
    def excluded_idx(self):
        if self._excluded_idx is None:
            # initialize self._excluded_idx
            __ = self.idx2base_mapper
        return self._excluded_idx

    @property
    def idx2base_mapper(self):
        """
        Here you should implement a dictionary to map idx to the base value for delta machine learning
        """
        raise NotImplementedError

    @property
    def fl2base_mapper(self):
        """
        A helper mapper which map file_handle to the base value for delta machine learning.
        """
        raise NotImplementedError


class PLDeltaLearningDataset(DeltaLearningDataset):
    """
    Implement delta LinF9 machine learning model.
    """
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, **kwargs)

    @property
    def fl2base_mapper(self):
        # file handle to base value (lin_f9 predicted value) mapper
        # eg: "CSAR_decoy_part1.1a8i.1a8i_decoys_108" -> 3.1433347205
        if self._fl2base_mapper is None:
            info_csv = osp.join(self.root, "..", "pl_xgb_lin_f9.csv")
            info_mapper = pd.read_csv(info_csv).set_index("lig_name")["lin_f9"].to_dict()
            for key in list(info_mapper.keys()):
                if np.isnan(info_mapper[key]):
                    del info_mapper[key]
            self._fl2base_mapper = info_mapper
        return self._fl2base_mapper

    @property
    def idx2base_mapper(self):
        if self._idx2base_mapper is None:
            mapper = {}
            self._excluded_idx = set()
            for i, (source, protein_file, ligand_file) in enumerate(zip(self.data.source, self.data.protein_file, self.data.ligand_file)):
                fl = f"{source}.{osp.basename(protein_file[0]).split('.')[0]}.{osp.basename(ligand_file[0]).split('.')[0]}"
                if fl not in self.fl2base_mapper.keys():
                    self._excluded_idx.add(i)
                else:
                    mapper[i] = self.fl2base_mapper[fl]
            self._idx2base_mapper = mapper
        return self._idx2base_mapper

class CASFSoringDeltaLearningDS(DeltaLearningDataset):
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, **kwargs)

    @property
    def fl2base_mapper(self):
        if self._fl2base_mapper is None:
            # TODO: put this file to a better location
            info_mapper = pd.read_csv("/scratch/sx801/scripts/delta_LinF9_XGB/performance/pred_score/pred.csv").set_index("pdb")["lin_f9"].to_dict()
            self._fl2base_mapper = info_mapper
        return self._fl2base_mapper

    @property
    def idx2base_mapper(self):
        if self._idx2base_mapper is None:
            mapper = {}
            self._excluded_idx = set()
            for i, lig_file in enumerate(self.data.ligand_file):
                fl = lig_file[0].split("_")[0]
                mapper[i] = self.fl2base_mapper[fl]
            self._idx2base_mapper = mapper
        return self._idx2base_mapper

class CASFDockingDeltaLearningDS(DeltaLearningDataset):
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, **kwargs)

    @property
    def fl2base_mapper(self):
        if self._fl2base_mapper is None:
            # The files are located in the singularity container: /scratch/sx801/data/CASF-2016-cyang.sqf
            info_mapper = {}
            for info_csv in glob("/CASF-2016-cyang/power_docking/examples/Lin_v9/*.dat"):
                this_mapper = pd.read_csv(info_csv, sep=" ").set_index("#code")["score"].to_dict()
                info_mapper.update(this_mapper)
            self._fl2base_mapper = info_mapper
        return self._fl2base_mapper

    @property
    def idx2base_mapper(self):
        if self._idx2base_mapper is None:
            mapper = {}
            self._excluded_idx = set()
            for i, lig_file in enumerate(self.data.ligand_file):
                fl = lig_file[0]
                mapper[i] = self.fl2base_mapper[fl]
            self._idx2base_mapper = mapper
        return self._idx2base_mapper

class CASFScreeningDeltaLearningDS(VSDummyIMDataset, DeltaLearningDataset):
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, **kwargs)

    @property
    def fl2base_mapper(self):
        if self._fl2base_mapper is None:
            # The files are located in the singularity container: /scratch/sx801/data/CASF-2016-cyang.sqf
            info_mapper = {}
            target_pdb = osp.basename(self.processed_file_names[0]).split(".")[0]
            info_csv = f"/CASF-2016-cyang/power_screening/examples/Lin_F9/{target_pdb}_score.dat"
            this_mapper = pd.read_csv(info_csv, sep=" ").set_index("#code_ligand_num")["score"].to_dict()
            info_mapper.update(this_mapper)
            self._fl2base_mapper = info_mapper
        return self._fl2base_mapper

    @property
    def idx2base_mapper(self):
        if self._idx2base_mapper is None:
            mapper = {}
            self._excluded_idx = set()
            for i, lig_file in enumerate(self.data.ligand_file):
                fl = lig_file[0]
                mapper[i] = self.fl2base_mapper[fl]
            self._idx2base_mapper = mapper
        return self._idx2base_mapper
