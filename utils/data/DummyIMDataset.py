from collections import defaultdict
import copy
from glob import glob
import logging
import os.path as osp
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import copy
import pandas as pd
import torch
from torch.utils.data import ConcatDataset
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data, HeteroData
from tqdm import tqdm
from ase.units import Hartree, eV
from utils.data.DataPreprocessor import DataPreprocessor
from utils.data.MyData import MyData
from utils.data.data_utils import get_lig_natom, get_prot_natom

from utils.data.rmsd_info_query import RMSD_Query
from utils.utils_functions import floating_type, lazy_property

hartree2ev = Hartree / eV


class DummyIMDataset(InMemoryDataset):
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, collate=False,
                 **kwargs):
        self.sub_ref = sub_ref
        self.dataset_name = dataset_name
        self.split = split
        self.cfg: dict = config_args
        super().__init__(data_root, None, None)
        from utils.data.DataPostProcessor import DataPostProcessor
        self.pre_processor = DataPreprocessor(config_args)
        self.post_processor = DataPostProcessor(config_args)
        if collate:
            self.data, self.slices = InMemoryDataset.collate(torch.load(self.processed_paths[0]))
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])
        if not isinstance(self.data, (MyData, HeteroData)):
            self.data = MyData.from_data(self.data)
        self.infuse_atom_mol_batch()
        self.infuse_sample_id()
        self.infuse_pdb()
        self.train_index, self.val_index, self.test_index = None, None, None
        self.parse_split_index(split)

        if self.sub_ref:
            warnings.warn("sub_ref is deprecated")
            preprocess_dataset(osp.join(osp.dirname(__file__), "GaussUtils"), self, convert_unit)

        if config_args is None:
            return
        
        if config_args["debug_mode"]:
            self.debug_mode_modify(config_args)
        # training a DiffDock confidence model-like classfication model
        self.rmsd_threshold: Optional[float] = config_args["rmsd_threshold"]
        self.no_pkd_score: bool = config_args.get("no_pkd_score", False)
        # The RMSD of ligands after MMFF optimization. Used as a feature to predict pKd.
        self.infuse_rmsd_info: bool = (config_args["rmsd_csv"] is not None) and (not self.no_pkd_score)
        if self.infuse_rmsd_info:
            self.rmsd_query = RMSD_Query(config_args)

        if config_args["proc_lit_pcba"]:
            unwanted_key = ('protein', 'interaction', 'protein')
            del self.data[unwanted_key]
            if unwanted_key in self.slices: del self.slices[unwanted_key]

    def infuse_pdb(self):
        if hasattr(self.data, "pdb"):
            return
        # LIT-PCBA
        if hasattr(self.data, "file_handle_ranked"):
            pdb_list = [osp.basename(f[0]).split(".")[0].split("_")[-1] for f in self.data.file_handle_ranked]
            self.data.pdb = pdb_list
            self.slices["pdb"] = copy.copy(self.slices["file_handle_ranked"])
            return
        if not hasattr(self.data, "protein_file"):
            self.data.pdb = ["" for __ in self.data.file_handle]
            self.slices["pdb"] = copy.copy(self.slices["file_handle"])
            return
        
        assert hasattr(self.data, "protein_file")
        pdb_list = [osp.basename(f).split(".")[0].split("_")[0] for f in self.data.protein_file]
        self.data.pdb = pdb_list
        self.slices["pdb"] = copy.copy(self.slices["protein_file"])

    def debug_mode_modify(self, cfg: dict):
        # In debug mode, the data set tries to load a minimal amount of data to reduce resource usage
        # By default, 1000 trainning examples and 100 validation examples are randomly selected from
        # the original data set.
        logging.info(("***********DEBUG MODE ON, Result not trustworthy***************"))
        n_train: int = cfg["debug_mode_n_train"]
        n_val: int = cfg["debug_mode_n_val"]
        if self.train_index is not None:
            perm = torch.randperm(len(self.train_index))
            self.train_index = self.train_index[perm[:n_train]]
        if self.val_index is not None and n_val > 0:
            perm = torch.randperm(len(self.val_index))
            self.val_index = self.val_index[perm[:n_val]]

    def infuse_atom_mol_batch(self):
        if hasattr(self.data, "atom_mol_batch"):
            return
        if "Z" not in self.slices:
            return
        # batch index mapping atom to molecules. Here initialized as 0,
        # but will be assigned by the dataloader collate function
        self.data.atom_mol_batch = torch.zeros(self.data.Z.shape[0]).long()
        self.slices["atom_mol_batch"] = copy.copy(self.slices["Z"])
    
    def infuse_sample_id(self):
        # mol index
        if hasattr(self.data, "sample_id"):
            return
        
        try_keys: List[str] = ["N", "N_aa_chain1", "pdb", "file_handle", "file_handle_ranked"]
        for cp_key in try_keys:
            if cp_key in self.slices: break
        cp_prop = getattr(self.data, cp_key)
        size = cp_prop.shape[0] if isinstance(cp_prop, torch.Tensor) else len(cp_prop)
        self.data.sample_id = torch.arange(size).long()
        self.slices["sample_id"] = copy.copy(self.slices[cp_key])

    def parse_split_index(self, split: Optional[str]) -> None:
        if split is None:
            # if no split provided, default to all test
            self.test_index: torch.Tensor = torch.arange(len(self))
            return
        
        if split == "random_split":
            valid_size = int(self.cfg["valid_size"])
            train_index = torch.arange(len(self))
            np.random.seed(self.cfg["split_seed"])
            perm_matrix = np.random.permutation(len(train_index))
            self.train_index = train_index[perm_matrix[:-valid_size]]
            self.val_index = train_index[perm_matrix[-valid_size:]]
            return

        split_data = torch.load(osp.join(self.processed_dir, self.split))
        if split_data["test_index"] is None:
            self.test_index = None
        else:
            self.test_index = torch.as_tensor(split_data["test_index"])

        # A hell of logic
        rand_val_index = False
        if ("valid_index" not in split_data) and ("val_index" not in split_data):
            rand_val_index = True
        elif "val_index" in split_data and split_data["val_index"] is None:
            rand_val_index = True
        # And even more
        if rand_val_index and split_data["train_index"] is None:
            rand_val_index = False
        if rand_val_index:
            valid_size = int(self.cfg["valid_size"])
            warnings.warn("You are randomly generating valid set from training set")
            train_index = torch.as_tensor(split_data["train_index"]).long()
            np.random.seed(2333)
            perm_matrix = np.random.permutation(len(train_index))
            self.train_index = train_index[perm_matrix[:-valid_size]]
            self.val_index = train_index[perm_matrix[-valid_size:]]
            return

        if split_data["train_index"] is not None:
            self.train_index = torch.as_tensor(split_data["train_index"]).long()
        else:
            self.train_index = None

        if split_data["test_index"] is not None:
            self.test_index = torch.as_tensor(split_data["test_index"]).long()
        else:
            self.test_index = None

        for name in ["val_index", "valid_index"]:
            if name not in split_data.keys():
                continue

            if split_data[name] is not None:
                self.val_index = torch.as_tensor(split_data[name]).long()
            else:
                self.val_index = None


    def get(self, idx: int, process=True) -> Data:
        res = super().get(idx)
        res = self.pre_processor(res, idx)
        if self.cfg is None:
            return res

        if self.rmsd_threshold is not None:
            res.within_cutoff = (res.rmsd < self.rmsd_threshold).long()

        if process:
            res = self.post_processor(res, idx)
        res = self.try_infuse_rmsd_info(res)
        return res
    
    def try_infuse_rmsd_info(self, data):
        if not self.infuse_rmsd_info:
            return data
        query: str = getattr(data, self.cfg["lig_identifier_src"])
        # "short_name" is only during testing on external test sets such as CASF-2016 and LIT-PCBA
        short_name: str = self.cfg["short_name"] if "short_name" in self.cfg else None
        if short_name == "casf2016-docking":
            query = data.ligand_file
            if isinstance(query, list): query = query[0]
        if short_name == "casf2016-screening":
            ligand_file = data.ligand_file
            if isinstance(ligand_file, list): ligand_file = ligand_file[0]
            query = f"{data.pdb}_{ligand_file}"
        rmsd_info: float = self.rmsd_query.query_rmsd(query)
        data.rmsd = torch.as_tensor([rmsd_info]).type(floating_type)
        return data

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name]

    def download(self):
        pass

    def process(self):
        pass


class AuxPropDataset(DummyIMDataset):
    def __init__(self, data_root, config_args, **kwargs):
        super().__init__(data_root=data_root, config_args=config_args, **kwargs)

        # prepare pdb mapper information which is needed to fetch SASA information
        idx2pdb_mapper = {}
        # prepare ligand file mapper which is the ligand identifier for pre-computed atom-prop information
        idx2lig_mapper = {}
        is_merck_fep: bool = "merck.fep" in self.dataset_name
        for i in tqdm(range(len(self)), desc="idx2pdb", total=len(self)):
            this_d = super().get(i, process=False)
            lig_identifier = getattr(this_d, self.cfg["lig_identifier_dst"], None)
            if isinstance(lig_identifier, list): 
                lig_identifier = lig_identifier[0]
            idx2lig_mapper[i] = self._lig_file_hash(lig_identifier)

            if is_merck_fep:
                idx2pdb_mapper[i] = this_d.target
                continue
            # prefix used by some of the datasets
            prefix = ""
            if hasattr(this_d, "source"):
                prefix = this_d.source + "."

            # use provided PDB
            if hasattr(this_d, "pdb"):
                idx2pdb_mapper[i] = prefix + this_d.pdb
                continue

            # if "pdb" not provided, infer from files
            if hasattr(this_d, "protein_file"):
                src_file = this_d.protein_file[0]
            else:
                src_file = this_d.ligand_file[0]
            if src_file == "":
                # This is a special logic when I process the dataset. 
                # I set the src_file to "" when I the protein information is the same as previous one.
                # the purpose is to save disk and memory
                idx2pdb_mapper[i] = idx2pdb_mapper[i-1]
            else:
                idx2pdb_mapper[i] = prefix + osp.basename(src_file).split(".")[0].split("_")[0]
        self.idx2pdb_mapper = idx2pdb_mapper
        self.idx2lig_mapper = idx2lig_mapper

        # if the --prot_info_ds argument is provided, the protein information will be added on the fly.
        # it is useful in the virtual screening dataset when you have multiple ligand versus one target,
        # you do not want to save protein geometries multiple times.
        self.prot_info_mapper = None
        self.prot_infuse_keys = set(["N_prot", "PP_min_dist_oneway_dist", "PP_min_dist_oneway_edge_index"])
        if config_args["prot_info_ds"] is not None:
            prot_info_mapper = {}
            # ensuring proper RMSD info
            prot_info_ds_args: dict = copy.deepcopy(config_args)
            if "short_name" in prot_info_ds_args: del prot_info_ds_args["short_name"]
            prot_info_ds_args["rmsd_csv"] = "/vast/sx801/geometries/PL_physics_infusion/PDBBind2020_OG/info/pdbbind2020_og.rmsd.csv"
            prot_info_ds =  DummyIMDataset(data_root, config_args["prot_info_ds"], prot_info_ds_args)
            for i in range(len(prot_info_ds)):
                this_d = prot_info_ds[i]
                this_pdb = osp.basename(this_d.protein_file[0]).split("_")[0]
                if hasattr(this_d, "pdb"):
                    this_pdb = this_d.pdb
                prot_info_mapper[this_pdb] = this_d
            self.prot_info_mapper = prot_info_mapper

        self.is_testing = (self.train_index is None)
        # use explicit ds argument will trigger atom_prop infusion
        self.mdn_w_lig_atom_props = config_args["mdn_w_lig_atom_props"]
        self.mdn_w_prot_sasa = config_args["mdn_w_prot_sasa"]
        self.want_atom_prop: bool = (config_args["atom_prop_ds"] is not None) and (not self.is_testing and self.mdn_w_lig_atom_props > 0.)
        self.want_prot_prop: bool = (not self.is_testing and self.mdn_w_prot_sasa > 0.)
        self.want_mol_prop: bool = self.cfg["precomputed_mol_prop"] and not (self.no_pkd_score)
        # Special logic: when testing on external test sets ("short_name" is available in config)
        # the pre-computed mol_prop is no longer useful and want_mol_prop has to be disabled.
        # Instead, mol_prop will be computed on-the-fly in MPNNPairedPropLayer
        if "short_name" in self.cfg and self.cfg["short_name"] == "casf2016-scoring":
            self.want_mol_prop = False

        self.delta_learning_pkd = config_args["delta_learning_pkd"]

        if self.want_prot_prop:
            # prepare SASA mapper information
            prot_sasa_ds = DummyIMDataset(data_root, "PDBind_v2020OG_prot.martini_sasa.pyg", config_args)
            prot_sasa_mapper = {}
            for i in range(len(prot_sasa_ds)):
                this_d = prot_sasa_ds.get(i, process=False)
                this_sasa = this_d.martini_sasa
                this_pdb = this_d.pdb
                prot_sasa_mapper[this_pdb] = this_sasa
            self.prot_sasa_mapper = prot_sasa_mapper
        
        # track unavailable mol_props
        self.n_molprop_queried = 0
        self.n_molprop_missing = 0

    @lazy_property
    def atom_prop_mapper(self) -> Dict[str, torch.Tensor]:
        if not self.want_atom_prop:
            return None
        atom_props_ds = self.aquire_atom_prop_ds()
        atom_prop_mapper = {}
        for i in range(len(atom_props_ds)):
            this_d = atom_props_ds.get(i, process=False)
            this_atom_prop = this_d.atom_prop
            idenfifier: str = getattr(this_d, self.cfg["lig_identifier_src"])
            if self.cfg["lig_identifier_src"] != "pdb": idenfifier = idenfifier[0]
            assert isinstance(idenfifier, str), idenfifier.__class__
            idenfifier = self._lig_file_hash(idenfifier)
            atom_prop_mapper[idenfifier] = this_atom_prop
        return atom_prop_mapper

    @lazy_property
    def mol_prop_mapper(self) -> Dict[str, torch.Tensor]:
        if not self.want_mol_prop:
            return None
        atom_props_ds_list = self.aquire_atom_prop_ds()
        mol_prop_mapper = {}
        lig_identifier = self.cfg["lig_identifier_src"]
        if "merck.fep" in self.dataset_name:
            lig_identifier = "file_handle_ranked"
        for atom_props_ds in atom_props_ds_list:
            for i in range(len(atom_props_ds)):
                this_d = atom_props_ds.get(i, process=False)
                this_mol_prop = this_d.mol_prop
                idenfifier: str = getattr(this_d, lig_identifier)
                if isinstance(idenfifier, list): idenfifier: str = idenfifier[0]
                assert isinstance(idenfifier, str), idenfifier.__class__
                idenfifier = self._lig_file_hash(idenfifier)
                mol_prop_mapper[idenfifier] = this_mol_prop
        return mol_prop_mapper

    def aquire_atom_prop_ds(self):        
        if self.cfg["atom_prop_ds"] is not None:
            atom_prop_ds_list: List[str] = glob(osp.join(self.root, "processed", self.cfg["atom_prop_ds"]))
            assert len(atom_prop_ds_list) > 0
            atom_prop_ds_list = [osp.relpath(f, osp.join(self.root, "processed")) for f in atom_prop_ds_list]
            prop_ds_args: dict = copy.deepcopy(self.cfg)
            prop_ds_args["precomputed_mol_prop"] = False
            if "short_name" in prop_ds_args: del prop_ds_args["short_name"]
            prop_ds_args["rmsd_csv"] = None
            prop_ds_args["proc_lit_pcba"] = False
            atom_props_ds_list = [DummyIMDataset(self.root, f, prop_ds_args) for f in atom_prop_ds_list]
            # atom_props_ds = ConcatDataset(atom_props_ds_list)
            return atom_props_ds_list
        
        # legacy ways of infering atom-prop dataset from the training dataset
        # prepare atom_prop mapper information
        if self.dataset_name in ["PDBind_v2020OG_prot.multires.lig.polar_proton.voronoi1.pyg", "PDBind_v2020OG_prot.dry_atom_pad.lig.polar_proton.pyg", 
                            "PDBind_v2020OG_prot.implicit.lig.polar_proton.min_dist.pyg"]:
            atom_props_ds = DummyIMDataset(self.root, "PDBind_v2020OG_ligatomprop_frag20sol_012_ens.pyg", self.cfg)
        elif self.dataset_name.startswith("PDBind_v2020OG_DiffDock_LinF9_prot."):
            atom_props_ds = DummyIMDataset(self.root, "PDBind_v2020OG_DiffDock_LinF9_ligatomprop_frag20sol_012_ens.pyg", self.cfg)
        else:
            assert self.dataset_name.startswith("PDBind_v2020OG_LinF9_prot."), self.dataset_name
            atom_props_ds = DummyIMDataset(self.root, "PDBind_v2020OG_LinF9_ligatomprop_frag20sol_012_ens.pyg", self.cfg)
        return [atom_props_ds]

    def get(self, idx: int, process=True) -> Data:
        # process it after infusion
        res = super().get(idx, process=False)

        this_lig = self.idx2lig_mapper[idx]
        this_pdb = self.idx2pdb_mapper[idx]

        res = self.add_prot_info(this_pdb, res)
        if process:
            res = self.post_processor(res, idx)

        if self.want_mol_prop:
            res = self.infuse_mol_props(this_lig, res)

        # aux tasks are only needed during training.
        if not self.want_atom_prop and not self.want_prot_prop:
            return res

        if self.want_atom_prop:
            res = self.infuse_atom_props(this_lig, res)
        if self.want_prot_prop:
            res = self.infuse_prot_props(this_pdb, res)
        return res
    
    def add_prot_info(self, pdb: str, res: Data):
        # reuse protein info during docking and screening.
        # Since the protein structure is the same for all ligands/decoys, duplicate information
        # is removed during data preprocessing and the protein information is added on the fly
        if self.prot_info_mapper is None:
            return res
        prot_info_d = self.prot_info_mapper[pdb]
        # for hetero data
        if isinstance(res, HeteroData):
            wanted_keys = ["protein", ("protein", "interaction", "protein"),
                           ("ion", "interaction", "protein")]
            assert isinstance(prot_info_d, HeteroData), prot_info_d.__class__
            for key in wanted_keys:
                for storage_key in prot_info_d[key].keys():
                    setattr(res[key], storage_key, getattr(prot_info_d[key], storage_key))
            return res
        # for homogeneous data
        for key in self.prot_infuse_keys:
            if not hasattr(prot_info_d, key):
                continue
            setattr(res, key, getattr(prot_info_d, key))
        return res
    
    def infuse_prot_props(self, this_pdb, res: Data):
        N_p = get_prot_natom(res)
        if this_pdb in self.prot_sasa_mapper:
            res.martini_sasa = self.prot_sasa_mapper[this_pdb]
            nan_mask = res.martini_sasa.isnan()
            res.mdn_w_prot_sasa = torch.zeros(N_p).type(floating_type).fill_(self.mdn_w_prot_sasa)
            res.martini_sasa[nan_mask] = 0.
            res.mdn_w_prot_sasa[nan_mask] = 0.
        else:
            res.martini_sasa = torch.zeros(N_p).type(floating_type)
            res.mdn_w_prot_sasa = torch.zeros(N_p).type(floating_type).fill_(0.)

        return res
    
    def infuse_atom_props(self, this_lig, res: Data):
        N_l = get_lig_natom(res)
        if this_lig in self.atom_prop_mapper:
            res.atom_prop = self.atom_prop_mapper[this_lig]
            res.mdn_w_lig_atom_props = torch.zeros(N_l).type(floating_type).fill_(self.mdn_w_lig_atom_props)
        else:
            # delta machine learning with missing data is not implemented yet
            assert not self.delta_learning_pkd
            res.atom_prop = torch.zeros(N_l, 3).type(floating_type)
            res.mdn_w_lig_atom_props = torch.zeros(N_l).type(floating_type).fill_(0.)
        if self.delta_learning_pkd:
            # base_value will be picked up by LossFn to calculate corrected prediction
            # we want pred minus water_energy to be the correct energy
            res.base_value = - res.atom_prop[:, 1].sum(dim=0)
        assert res.atom_prop.shape[0] == N_l, f"{res.atom_prop.shape}; {N_l}"
        return res
    
    def infuse_mol_props(self, this_lig, res):
        self.n_molprop_queried += 1
        if this_lig not in self.mol_prop_mapper:
            self.n_molprop_missing += 1
            res.mol_prop =torch.as_tensor([0., 0., 0.])
            return res
        res.mol_prop = self.mol_prop_mapper[this_lig]

        if self.n_molprop_queried % 1000 == 0:
            assert 1. * self.n_molprop_missing / self.n_molprop_queried < 0.1, "Too many missing molprops"
        return res
    
    def _lig_file_hash(self, lig_file: Optional[str]):
        if lig_file is None: return ""
        if "/" not in lig_file: return lig_file
        # As of 11/22 you can use any identifier, like: pdb_id
        # the hashing only works for ligand files
        if self.cfg["lig_identifier_src"] != "ligand_file": return lig_file

        base = osp.basename(lig_file)
        dir1 = osp.basename(osp.dirname(lig_file))
        return f"{dir1}-{base}"


class VSDummyIMDataset(DummyIMDataset):
    """
    Dataset for virtural screening. Store protein coordinates only at d[0] to reduce disk space.
    """
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, ref=False, **kwargs):
        self.ref = ref
        self._has_prot_context = None
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, collate, **kwargs)

        self.modify_test_set()

        # if we use reference ligand, the protein will be pre-cut and therefore on-the-fly cutting is diabled
        if ref:
            self.post_processor.cut_protein = False
            self._ref_cut_protein_data = None

        self.request_voronoi = self.post_processor.request_voronoi

    def get(self, idx: int, process=True) -> Data:
        if not process:
            return super().get(idx, process=process)
        
        # protein information is stored in d0
        # the i_th data only contains ligand information
        if idx in self.has_prot_context:
            assert not self.ref
            return super().get(idx)

        d0, di = self.get_d0_di(idx)

        if hasattr(di, "LIGAND_edge_index"):
            d0.LIGAND_edge_index = di.LIGAND_edge_index

        N_l = di.N_l
        N_l_org = d0.N_l
        # the mol_type of ligand is 1, this mask can be used to remove ligand atoms
        # the remaining atoms are protein (and beta atoms if available)
        rm_ligand_mask = (d0.mol_type != 1)
        di_ligand_mask = (di.mol_type == 1)
        # number of atoms to be added from d0
        N_d0 = rm_ligand_mask.sum()
        # concatenate the ligand information with the context information (protein and beta atoms)
        d0.R = torch.concat([di.R[di_ligand_mask, :], d0.R[rm_ligand_mask, :]])
        d0.mol_type = torch.concat([di.mol_type[di_ligand_mask], d0.mol_type[rm_ligand_mask]])
        d0.Z = torch.concat([di.Z[di_ligand_mask], d0.Z[rm_ligand_mask]])
        d0.N = N_d0 + N_l
        d0.N_l = N_l
        d0.ligand_file = di.ligand_file
        if self.request_voronoi:
            d0.LIGAND_Voronoi1_edge_index = di.LIGAND_Voronoi1_edge_index
            d0.PL_Voronoi1_edge_index = di.PL_Voronoi1_edge_index
            # correct the index number for protein edge index due to the difference in ligand atoms
            protein_correct = N_l - N_l_org
            d0.PROTEIN_Voronoi1_edge_index = d0.PROTEIN_Voronoi1_edge_index.long() + protein_correct
        res = self.post_processor(d0, idx)
        return res
    
    def modify_test_set(self):
        if self.ref:
            # the 0th data is the reference ligand, which is only used to calculate protein pocket
            self.test_index = self.test_index[1:]

    @property
    def has_prot_context(self):
        if self._has_prot_context is None:
            self._has_prot_context = set()
            if not self.ref:
                self._has_prot_context = set([0])
        return self._has_prot_context

    def get_d0_di(self, idx):
        d0 = super().get(0, process=False)

        # use the cut structure instead
        if self.ref:
            if self._ref_cut_protein_data is None:
                self._ref_cut_protein_data = self.post_processor.do_cut_protein(d0)
            d0 = self._ref_cut_protein_data.clone()
        di = super().get(idx, process=False)
        return d0, di


class VSPointerDummyIMDataset(VSDummyIMDataset):
    """
    Like VSDummyIMDataset, we want to reuse protein coordinate. This time, we store multiple protein structures in one dataset.
    To locate them, we use a ref_pointer to point the index of the stored protein struture and
     is_decoy to indicate if this entry has only ligand strure or not.
    This dataset is only used for some of the docking score calculation.
    """
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, ref=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, ref, **kwargs)

        self._ref_cut_ckpt = {}

    def modify_test_set(self):
        if self.ref:
            # the 0th data is the reference ligand, which is only used to calculate protein pocket
            self.test_index = self.test_index[self.data.is_decoy]

    @property    
    def has_prot_context(self):
        if self._has_prot_context is None:
            self._has_prot_context = set()
            if not self.ref:
                self._has_prot_context = set(self.data.ref_pointer.numpy().tolist())
        return self._has_prot_context

    def get_d0_di(self, idx):
        di = super(VSDummyIMDataset, self).get(idx, process=False)
        ref_idx = di.ref_pointer.item()
        d0 = super(VSDummyIMDataset, self).get(ref_idx, process=False)

        # use the cut structure instead
        if self.ref:
            if ref_idx not in self._ref_cut_ckpt.keys():
                self._ref_cut_ckpt[ref_idx] = self.post_processor.do_cut_protein(d0)
            d0 = self._ref_cut_ckpt[ref_idx].clone()
        return d0, di


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

