"""
Custom made protein-ligand data set for ESM-GearNet-NMDN model
The data set contains protein information processed for ESM-GearNet,
as well as ligand information for sPhysNet and NMDN model.
"""
from collections import defaultdict
import logging
import os
import os.path as osp
import csv
import glob
import pickle
import prody

import torch
from torch.utils import data as torch_data
from typing import List, Dict, Set
from easydict import EasyDict
from tempfile import TemporaryDirectory
import numpy as np

from torchdrug import data, utils, core
from torchdrug.core import Registry as R
from torchdrug.data.protein import Protein
from tqdm import tqdm

from utils.data.DummyIMDataset import DummyIMDataset
from utils.utils_functions import lazy_property
from Networks.esm_gearnet.model_wrapper import ESMGearnet
# /scratch/sx801/scripts/ESM-GearNet
from util import load_config

@R.register("datasets.ESMGearNetProtLig")
class ESMGearNetProtLig(data.ProteinDataset):
    def __init__(self, array_id: int=None, **kwargs) -> None:
        super().__init__()
        self.array_id = array_id
        # load configurations from original ESM-GearNet Config
        esm_gearnet_cofig_file: str = osp.join(osp.dirname(__file__), "esm_gearnet.yaml")
        egearnet_cfg: EasyDict = load_config(esm_gearnet_cofig_file, {})
        ds_cfg: EasyDict = egearnet_cfg.dataset
        egearnet_kwargs: dict = {}
        for key in ds_cfg:
            if key == "class" or key == "path" or key == "test_cutoff":
                continue
            if key == "transform":
                egearnet_kwargs[key] = core.Configurable.load_config_dict(ds_cfg[key])
                continue
            egearnet_kwargs[key] = ds_cfg[key]
        self.egearnet_kwargs = egearnet_kwargs

        # DummyDataset for ligand information
        lig_ds = DummyIMDataset(**kwargs)
        self.lig_ds = lig_ds
        # copy variables
        var_names = ["train_index", "val_index", "test_index", "processed_file_names"]
        for key in var_names:
            setattr(self, key, getattr(lig_ds, key))
        
        # Caching file names
        dataset_name: str = lig_ds.dataset_name.split(".pyg")[0]
        if dataset_name.startswith("casf"):
            dataset_name = "PBind2020OG.hetero.polar.polar.implicit.min_dist"
            dataset_name = "casf-scoring.prot.polar.lig.polar.implicit.min_dist"
        processed_root: str = lig_ds.processed_dir
        processed_pkl: str = osp.join(processed_root, f"{dataset_name}.esm_gearnet.pkl.gz")
        self.dataset_name = dataset_name
        self.processed_pkl = processed_pkl

        # preprocessing
        pdb_files: List[str] = [lig_ds[i].protein_file for i in range(len(lig_ds))]
        if isinstance(pdb_files[0], list):
            pdb_files: List[str] = [i[0] for i in pdb_files]
        pdb_ids: List[str] = [osp.basename(f).split(".")[0].split("_")[0] for f in pdb_files]
        self.pdb_ids: List[str] = pdb_ids
        del pdb_files

        if osp.exists(processed_pkl):
            self.load_pickle(processed_pkl, verbose=True, **egearnet_kwargs)
            return self.post_init()

        self.preprocess_hpc_array()

    def preprocess_hpc_array(self):
        # For large data set (>10000), preprocessing takes a lot of time
        # the following code use a HPC array to process with multiple CPUs.
        # number of CPUs for the job
        N_ARRAYS = 1

        lig_ds = self.lig_ds
        array_id = self.array_id
        egearnet_kwargs = self.egearnet_kwargs

        if array_id is not None:
            # To speed up the data set preprocessing, I used an array of 30 CPUs to prcess simoutaneously
            processed_root: str = osp.join("/vast/sx801/single_pygs", f"{self.dataset_name}.esm_gearnet")
            os.makedirs(processed_root, exist_ok=True)
            self.processed_pkl: str = osp.join(processed_root, f"array.{array_id}.pkl.gz")
        
        if array_id is not None:
            # preprocess array
            pdb_chunks = np.array_split(self.pdb_ids, N_ARRAYS)
            tempdir = TemporaryDirectory()
            pdb_ids = pdb_chunks[array_id]
            pdb_files = [self.ds_reader.pdb2prot_noh(pdb) for pdb in pdb_ids]
            pdb_files = [self.preprocess_pdb_file(f, tempdir.name) for f in pdb_files]
            self.load_pdbs(pdb_files, verbose=True, **egearnet_kwargs)
            self.save_pickle(self.processed_pkl, verbose=True)
            tempdir.cleanup()
            return self.post_init()
        
        assert array_id is None
        # collate
        processed_root: str = osp.join("/vast/sx801/single_pygs", f"{self.dataset_name}.esm_gearnet")
        array_pkls = [osp.join(processed_root, f"array.{i}.pkl.gz") for i in range(N_ARRAYS)]
        self.transform = egearnet_kwargs["transform"]
        self.lazy = False
        self.kwargs = egearnet_kwargs
        self.sequences = []
        self.pdb_files = []
        self.data = []
        for pkl_file in array_pkls:
            with utils.smart_open(pkl_file, "rb") as fin:
                num_sample = pickle.load(fin)
                indexes = range(num_sample)
                indexes = tqdm(indexes, "Loading %s" % osp.basename(pkl_file))
                for __ in indexes:
                    pdb_file, sequence, protein = pickle.load(fin)
                    self.sequences.append(sequence)
                    self.pdb_files.append(pdb_file)
                    self.data.append(protein)
        self.save_pickle(self.processed_pkl, verbose=True)
        self.post_init()

    def post_init(self):
        # modify train_index and val_index: remove indices without protein graphs (failed during preprocessing)
        for index_name in ["train_index", "val_index"]:
            idx = getattr(self, index_name)
            if idx is None: continue
            idx: List[int] = torch.as_tensor(idx).tolist()
            idx = [i for i in idx if self.pdb_ids[i] in self.pdb2data]
            setattr(self, index_name, idx)

        if self.test_index is not None:
            missing_pdbs: List[str] = []
            wanted_idx = []
            for i in self.test_index:
                if self.pdb_ids[i] not in self.pdb2data:
                    missing_pdbs.append(self.pdb_ids[i])
                else:
                    wanted_idx.append(i)
            if len(missing_pdbs) > 0:
                logging.warn(f"Missing {len(missing_pdbs)} pdbs: {missing_pdbs}")
            self.test_index = wanted_idx

    @lazy_property
    def pdb2data(self) -> Dict[str, Protein]:
        pdb2data = defaultdict(lambda: None)
        for pdb_file, data in zip(self.pdb_files, self.data):
            pdb_id: str = osp.basename(pdb_file).split(".")[0].split("_")[0]
            pdb2data[pdb_id] = data
        return pdb2data

    @lazy_property
    def ds_reader(self):
        from geometry_processors.pl_dataset.pdb2020_ds_reader import PDB2020DSReader
        from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
        if self.dataset_name.startswith("casf"):
            return CASF2016Reader("/CASF-2016-cyang")
        return PDB2020DSReader("/PDBBind2020_OG")

    def preprocess_pdb_file(self, pdb_file: str, sroot: str) -> str:
        # preprocess pdb file, including removing water molecules / ions
        ag = prody.parsePDB(pdb_file)
        protein = ag.protein.toAtomGroup()
        dst_file = osp.join(sroot, osp.basename(pdb_file))
        prody.writePDB(dst_file, protein)
        return dst_file

    def get_item(self, index) -> dict:
        # load protein information
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            pdb_id: str = self.pdb_ids[index]
            protein = self.pdb2data[pdb_id].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        # fake targets
        item["targets"] = torch.zeros([1, 1])
        # load ligand information
        item["ligand"] = self.lig_ds[index]
        # re-compute PL interaction
        return item
