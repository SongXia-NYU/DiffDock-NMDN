from copy import deepcopy
from typing import Union
from torch import LongTensor
import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from Networks.PairedPropLayers.PairedPropLayer import MPNNPairedPropLayer
from utils.configs import Config
from utils.data.data_utils import get_lig_batch
from utils.utils_functions import get_device


class HeteroPairedPropLayer(nn.Module):
    def __init__(self, cfg: Config, edge_name: str, activation: str, style="add") -> None:
        super().__init__()
        self.lig_prot_layer = MPNNPairedPropLayer(cfg, edge_name, activation, style)

        # registering ligand-metal layer
        lig_metal_cls = PrecumputedMetalLigandPairedPropLayer if cfg.model.mdn.metal_atom_embed_path is not None else MetalLigandPairedPropLayer
        lig_metal_cfg: Config = deepcopy(cfg)
        # disable RMSD and physical terms
        lig_metal_cfg.model.mdn["pkd_phys_terms"] = None
        lig_metal_cfg.data.pre_computed["rmsd_csv"] = None
        lig_metal_cfg.data.pre_computed["rmsd_expansion"] = None
        self.lig_metal_layer = lig_metal_cls(lig_metal_cfg, edge_name, activation, style)

        self.w_lig_metal: float = cfg.model.mdn.w_lig_metal

        # It is used during testing: disable this layer to only predict NMDN score
        self.no_pkd_score: bool = self.lig_prot_layer.no_pkd_score

    def forward(self, runtime_vars: dict):
        if self.no_pkd_score:
            return runtime_vars
        
        # compute protein-ligand interaction and record it
        runtime_vars = self.lig_prot_layer.forward(runtime_vars)
        lig_prot_prop: torch.Tensor = runtime_vars["pair_mol_prop"]

        # compute ligand-metal interaction and add PL interaction on to it
        data_batch: Union[Batch, dict] = runtime_vars["data_batch"]
        # data_batch is a dict when using ESM-GearNet
        if isinstance(data_batch, dict): data_batch = data_batch["ligand"]

        if data_batch["ion"].Z.shape[0] > 0:
            runtime_vars = self.lig_metal_layer.forward(runtime_vars)
            runtime_vars["pair_mol_prop"] = self.w_lig_metal * runtime_vars["pair_mol_prop"] + lig_prot_prop
        else:
            assert runtime_vars["kano_atom_embed"] is None

        if hasattr(data_batch, "linf9_score"):
            runtime_vars["pair_mol_prop"] = runtime_vars["pair_mol_prop"] + data_batch.linf9_score.view(-1, 1)

        return runtime_vars


class PrecumputedMetalLigandPairedPropLayer(MPNNPairedPropLayer):
    # metal ion embedding is pre-computed and accessed through self.metal_atom_embedding_layer
    def __init__(self, config_dict: dict, edge_name: str, activation: str, style="add") -> None:
        atom_embed_path = config_dict["metal_atom_embed_path"]
        # [-1, n_feature]
        self.metal_atom_embedding_layer = torch.load(atom_embed_path).to(get_device())
        
        n_slice = config_dict["metal_atom_embed_slice"]
        if n_slice is not None:
            self.metal_atom_embedding_layer = self.metal_atom_embedding_layer[:, :n_slice]

        super().__init__(config_dict, edge_name, activation, style)

    def get_prot_dim(self, config_dict: dict):
        # "protein" is actually metal ion here
        return self.metal_atom_embedding_layer.shape[-1]
    
    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        # data_batch is a dict when using ESM-GearNet
        if isinstance(data_batch, dict): data_batch = data_batch["ligand"]

        d0 = data_batch.get_example(0)

        assert isinstance(d0, HeteroData), data_batch.__class__
        return data_batch[("ligand", "interaction", "ion")].min_dist_edge_index, \
            data_batch[("ligand", "interaction", "ion")].min_dist
    
    def retrieve_edge_embed(self, data_batch: Batch, runtime_vars: dict, edge_index: LongTensor):
        # h_l: ligand embedding by order
        h_l = runtime_vars["vi"]
        # h_l_x is the embedding based on edges
        h_l_x = h_l[edge_index[0, :], :]

        z_m: LongTensor = data_batch["ion"].Z.view(-1)
        h_m = self.metal_atom_embedding_layer[z_m, :]
        h_m_x = h_m[edge_index[1, :], :]

        pair_batch = get_lig_batch(data_batch)[edge_index[0, :]]
        return h_l_x, h_m_x, pair_batch
    
class MetalLigandPairedPropLayer(MPNNPairedPropLayer):
    # metal ion embedding is from other DL models on-the-fly.
    # For example, using KANO to embed the metal ions. In this way, the KANO model will also be trained.
    def __init__(self, config_dict: dict, edge_name: str, activation: str, style="add") -> None:
        super().__init__(config_dict, edge_name, activation, style)

    def get_prot_dim(self, config_dict: dict) -> int:
        # "protein" is actually metal ion here
        # KANO's embedding dimension is 300
        return 300
    
    def retrieve_edge_embed(self, data_batch: Batch, runtime_vars: dict, edge_index: LongTensor):
        # h_l: ligand embedding by order
        h_l = runtime_vars["vi"]
        # h_l_x is the embedding based on edges
        h_l_x = h_l[edge_index[0, :], :]

        h_m = runtime_vars["kano_atom_embed"]
        h_m_x = h_m[edge_index[1, :], :]

        pair_batch = get_lig_batch(data_batch)[edge_index[0, :]]
        return h_l_x, h_m_x, pair_batch
    
    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        # data_batch is a dict when using ESM-GearNet
        if isinstance(data_batch, dict): data_batch = data_batch["ligand"]

        d0 = data_batch.get_example(0)

        assert isinstance(d0, HeteroData), data_batch.__class__
        return data_batch[("ligand", "interaction", "ion")].min_dist_edge_index, \
            data_batch[("ligand", "interaction", "ion")].min_dist
