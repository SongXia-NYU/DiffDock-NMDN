from argparse import Namespace
import os.path as osp
import pickle
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import logging
from chemprop_kano.features.featurization import ATOM_FDIM, BOND_FDIM, MolGraph, BatchMolGraph, get_bond_fdim

from chemprop_kano.models.cmpn import CMPNEncoder
from chemprop_kano.models.model import prompt_generator_output
from utils.configs import Config
from utils.utils_functions import get_device, lazy_property

# storing loaded KANO ds so it will not load twice.
loaded_kano_ds: Dict[str, Dict[str, MolGraph]] = {}

class KanoAtomEmbed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.kano_args = Namespace(hidden_size=300, bias=False, depth=3, dropout=0.0, undirected=False, atom_messages=False, features_only=False,
                              use_input_features=None, activation="ReLU", cuda=torch.cuda.is_available(), step="functional_prompt")
        fbond = get_bond_fdim(self.kano_args) + (not self.kano_args.atom_messages) * ATOM_FDIM
        self.encoder: CMPNEncoder = CMPNEncoder(self.kano_args, ATOM_FDIM, fbond)
        if self.kano_args.step == "functional_prompt":
            self.encoder.W_i_atom = prompt_generator_output(self.kano_args)(self.encoder.W_i_atom)

        self.load_kano_ds(cfg)

        # legacy behaviour
        self.folder_prefix = cfg.folder_prefix

        # load model checkpoint
        if cfg.model.kano.kano_ckpt is not None:
            logging.info(f"Loading KANO checkpoint from {cfg.model.kano.kano_ckpt}")
            self.load_state_dict(torch.load(cfg.model.kano.kano_ckpt, map_location=get_device()), strict=False)
        self.cfg = cfg

    def load_kano_ds(self, cfg: Config):
        # load KANO data from precomputed pickle file
        kano_ds_name: str = cfg.data.kano_ds
        ds_root: str = cfg.data.data_root
        kano_ds_path: str = osp.join(ds_root, "processed", kano_ds_name)
        kano_load_stype: str = "default"
        if kano_ds_name.startswith("casf-docking."):
            kano_load_stype: str = "casf-docking"
        if osp.isdir(kano_ds_path):
            kano_load_stype: str = "casf-screening"
        self.kano_load_style = kano_load_stype
        self.kano_ds_path = kano_ds_path
        self._kano_dynamic_loaded_ds: Tuple[Dict[str, MolGraph], str] = None

    def forward(self, runtime_vars: dict):
        mol_graphs: List[MolGraph] = self.get_mol_graphs(runtime_vars)
        if len(mol_graphs) == 0:
            # This happpens when the whole batch does not have a single metal ion
            runtime_vars["kano_atom_embed"] = None
            return runtime_vars

        mol_graph: BatchMolGraph = BatchMolGraph(mol_graphs, self.kano_args)

        atom_embed: torch.Tensor = self.encoder.forward_atom_embed(None, mol_graph, None)
        if self.folder_prefix != "exp_pl_396":
            # remove padding
            atom_embed = atom_embed[1:, :]
        runtime_vars["kano_atom_embed"] = atom_embed

        return runtime_vars
    
    def get_mol_graphs(self, runtime_vars: dict) -> List[MolGraph]:
        molid_name: str = "pdb"
        if self.cfg.data.kano_ds.startswith(("qm9", "esol")): molid_name = "mol_id"
        molid_list: List[str] = getattr(runtime_vars["data_batch"], molid_name)
        lig_files: List[List[str]] = runtime_vars["data_batch"].ligand_file
        lig_files: List[str] = [i[0] for i in lig_files]
        return [self.get_mol_graph(molid, lig_file) for molid, lig_file in zip(molid_list, lig_files)]

    def get_mol_graph(self, pdb: str, lig_file: str) -> MolGraph:
        # CASF docking uses the lig_file as keys: lig_file is like "1a30_66" indicating the ID
        if self.kano_load_style == "casf-docking":
            return self.kano_ds[lig_file]
        # CASF screening used lig_file as well.
        # Moreover, 57 pickle files are generated for each target. Only one is loaded each time to save memory.
        if self.kano_load_style == "casf-screening":
            if self._kano_dynamic_loaded_ds is None or self._kano_dynamic_loaded_ds[1] != pdb:
                with open(osp.join(self.kano_ds_path, f"{pdb}.pickle"), "rb") as f:
                    ds = pickle.load(f)
                self._kano_dynamic_loaded_ds = (ds, pdb)
            return self._kano_dynamic_loaded_ds[0][lig_file]
        # By default, the pdb is used as the key.
        return self.kano_ds[pdb]

    @lazy_property
    def kano_ds(self) -> Dict[str, MolGraph]:
        assert self.kano_load_style != "casf-screening"
        if self.kano_ds_path in loaded_kano_ds:
            return loaded_kano_ds[self.kano_ds_path]
        
        with open(self.kano_ds_path, "rb") as f:
            ds = pickle.load(f)
        loaded_kano_ds[self.kano_ds_path] = ds
        return ds

