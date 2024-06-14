import logging
import math
import os.path as osp
from glob import glob
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
from utils.LossFn import calculate_probablity, mdn_loss_fn
from utils.configs import Config
from utils.data.MyData import MyData
from utils.data.data_utils import get_lig_batch, get_lig_z

from utils.utils_functions import DistCoeCalculator, gaussian_rbf, get_device, info_resolver, lazy_property, softplus_inverse, floating_type

class MDNLayer(nn.Module):
    """
    A Mixture Density Network to learn the probability density distribution of the distance between
    ligand atom and protein (Martini bead or residues).
    """
    def __init__(self, mdn_edge_name, cfg: Config, dropout_rate=0.15, n_atom_types=95, martini2aa_pooling=False) -> None:
        super().__init__()
        loss_cfg = cfg.training.loss_fn
        model_cfg = cfg.model
        self.lig_atom_types: bool = loss_cfg["mdn_w_lig_atom_types"] > 0.
        self.prot_atom_types: bool = loss_cfg["mdn_w_prot_atom_types"] > 0.
        self.lig_atom_props: bool = loss_cfg["mdn_w_lig_atom_props"] > 0.
        self.prot_sasa: bool = loss_cfg["mdn_w_prot_sasa"] > 0.
        self.dropout_rate = dropout_rate
        self.cfg = cfg

        self.prot_dim = self.get_prot_dim()
        self.lig_dim = model_cfg["n_feature"]
        if "KANO" in model_cfg["modules"].split(" "):
            # KANO embed atoms into 300
            self.lig_dim = 300
        if "Comb-P-KANO" in model_cfg["modules"].split(" "):
            # KANO embed atoms into 300
            assert "KANO" in model_cfg["modules"].split(" "), model_cfg["modules"]
            self.lig_dim = 300 + model_cfg["n_feature"]
        if self.overwrite_lig_dim(): self.lig_dim = self.overwrite_lig_dim()

        self.cutoff_needed = max(model_cfg.mdn.mdn_threshold_train, model_cfg.mdn.mdn_threshold_eval)

        hidden_dim = self.gather_hidden_dim(cfg)
        assert hidden_dim is not None
        self.hidden_dim = hidden_dim
        self.register_mlp_layer()
        n_gaussians=cfg.model.mdn["n_mdn_gauss"]
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)

        if self.lig_atom_types:
            self.lig_atom_types_layer = nn.Linear(self.lig_dim, n_atom_types)
        if self.prot_atom_types:
            self.prot_atom_types_layer = nn.Linear(self.prot_dim, n_atom_types)
        if self.lig_atom_props:
            self.lig_atom_props_layer = nn.Linear(self.lig_dim, 3)
        if self.prot_sasa:
            self.prot_sasa_layer = nn.Linear(self.prot_dim, 1)

        self.mdn_edge_name = mdn_edge_name
        if self.mdn_edge_name == "None":
            self.mdn_edge_name = "PL_oneway"

    def get_prot_dim(self):
        return get_prot_dim(self.cfg)

    def gather_hidden_dim(self, cfg: Config):
        return cfg.model.mdn["n_mdn_hidden"]

    def register_mlp_layer(self):
        mlp_layers = [nn.Linear(self.lig_dim + self.prot_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)]
        for i in range(self.cfg.model.mdn["n_mdn_layers"]-1):
            mlp_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)])
        self.MLP = nn.Sequential(*mlp_layers)

    def forward(self, runtime_vars: dict):
        return self.forward_ext_prot_embed(runtime_vars)
    
    def forward_ext_prot_embed(self, runtime_vars: dict):
        """
        Use external protein embedding which is typically computed using ESM2.
        """
        h_l, h_p, h_l_i, h_p_j, pl_edge, pl_dist, data_batch, pair_batch = self.unpack_pl_info(runtime_vars)

        embeds = [h_l_i, h_p_j]
        pair_embeds = torch.cat(embeds, -1)
        pair_embeds = self.MLP(pair_embeds)

        # Outputs
        pi, sigma, mu = self.predict_gaussian_params(pair_embeds)
        out = {}
        out["pi"] = pi
        out["sigma"] = sigma
        out["mu"] = mu
        out["dist"] = pl_dist.detach()
        out["C_batch"] = pair_batch
        out["pl_edge_index_used"] = pl_edge
        runtime_vars.update(out)

        assert not self.prot_atom_types
        if self.lig_atom_types:
            lig_atom_types = self.lig_atom_types_layer(h_l)
            lig_atom_types_label = get_lig_z(data_batch)
            out["lig_atom_types"] = lig_atom_types
            out["lig_atom_types_label"] = lig_atom_types_label
        if self.prot_atom_types:
            # Save the problem for the future
            raise NotImplementedError
        if self.lig_atom_props:
            lig_atom_props = self.lig_atom_props_layer(h_l)
            out["lig_atom_props"] = lig_atom_props
        if self.prot_sasa:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_sasa = self.prot_sasa_layer(h_p)
            out["prot_sasa"] = prot_sasa
        return out

    def unpack_pl_info(self, runtime_vars: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Data, Tensor]:
        # return shape:
        # h_l: [N_lig, F_lig]   h_p: [N_prot, F_prot]
        # h_l_i: [N_pl, F_lig]  h_p_j: [N_pl, F_prot]
        # pl_edge: [2, N_pl]    pl_dist: [N_pl, 1]
        # data_batch: torch_geometric.Data
        data_batch = runtime_vars["data_batch"]
        # h_l: ligand embedding by order
        h_l = runtime_vars["vi"]
        # h_p: protein embedding by order
        h_p = runtime_vars["prot_embed"]

        pl_edge, pl_dist = self.retrieve_edge_info(data_batch)

        if self.cutoff_needed is not None:
            # only obtain needed pl_edges to avoid un-needed calculation
            pl_dist_mask = (pl_dist <= self.cutoff_needed)
            pl_edge = pl_edge[:, pl_dist_mask]
            pl_dist = pl_dist[pl_dist_mask]
        pl_dist = pl_dist.view(-1, 1)
        # h_l_i and h_p_j are the embedding based on edges
        h_l_i = h_l[pl_edge[0, :], :]
        h_p_j = h_p[pl_edge[1, :], :]
        pair_batch = get_lig_batch(data_batch)[pl_edge[0, :]]
        return h_l, h_p, h_l_i, h_p_j, pl_edge, pl_dist, data_batch, pair_batch

    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        if isinstance(data_batch, dict):
            return data_batch["esm-gearnet-pl_edge"], data_batch["esm-gearnet-pl_dist"]
        d0 = data_batch.get_example(0) if isinstance(data_batch, Batch) else data_batch
        if isinstance(d0, MyData):
            pl_edge = getattr(data_batch, f"{self.mdn_edge_name}_edge_index")
            pl_dist = getattr(data_batch, f"{self.mdn_edge_name}_dist")
        else:
            pl_edge = data_batch[("ligand", "interaction", "protein")].min_dist_edge_index
            pl_dist = data_batch[("ligand", "interaction", "protein")].min_dist
        return pl_edge, pl_dist
    
    def predict_gaussian_params(self, pair_embeds: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # readout layer to predict output Gaussian distribution parameters
        pi = F.softmax(self.z_pi(pair_embeds), -1)
        sigma = F.elu(self.z_sigma(pair_embeds))+1.1
        mu = F.elu(self.z_mu(pair_embeds))+1
        return pi, sigma, mu
    
    def overwrite_lig_dim(self) -> Optional[int]:
        return None
    
class GaussExpandLayer(nn.Module):
    def __init__(self, config_dict: Config, expansion_params: str = None) -> None:
        # avoid calling nn.Module twice when which seems to reset all parameters in the module
        # it is nessesary in the double inheritance in the class MDNPropLayer
        if '_buffers' not in self.__dict__:
            super().__init__()
        self.expansion_params: str = expansion_params
        self.cfg = config_dict
        self.register_expansion_params()

    def gaussian_dist_infuser(self, pair_dist: torch.Tensor):
        rbf = gaussian_rbf(pair_dist, self.centers, self.widths, self.cutoff, self.expansion_coe, self.linear_gaussian)
        return rbf
    
    def register_expansion_params(self):
        # register expansion params
        expansion_params = self.expansion_params if self.expansion_params else self.cfg.model.mdn.mdn_dist_expansion
        self.dist_infuser = None
        if expansion_params is None: return
        
        expansion_info = info_resolver(expansion_params)
        assert expansion_info['name'] == "gaussian", expansion_info

        # registering gaussian RBF parameters
        n_rbf = expansion_info['n']
        feature_dist = expansion_info['dist']
        feature_dist = torch.as_tensor(feature_dist).type(floating_type)
        self.register_parameter('cutoff', torch.nn.Parameter(feature_dist, False))
        expansion_coe = torch.as_tensor(expansion_info["coe"]).type(floating_type)
        self.register_parameter('expansion_coe', torch.nn.Parameter(expansion_coe, False))
        # Centers are params for Gaussian RBF expansion in PhysNet
        dens_min = expansion_info["dens_min"]
        centers = softplus_inverse(torch.linspace(math.exp(-dens_min), math.exp(-feature_dist * expansion_coe), n_rbf))
        centers = torch.nn.functional.softplus(centers)
        self.register_parameter('centers', torch.nn.Parameter(centers, False))

        # Widths are params for Gaussian RBF expansion in PhysNet
        widths = [softplus_inverse((0.5 / ((1.0 - torch.exp(-feature_dist)) / n_rbf)) ** 2)] * n_rbf
        widths = torch.as_tensor(widths).type(floating_type)
        widths = torch.nn.functional.softplus(widths)
        self.register_parameter('widths', torch.nn.Parameter(widths, False))
        self.linear_gaussian = expansion_info["linear"]

        self.dist_infuser = self.gaussian_dist_infuser

    @lazy_property
    def n_rbf(self) -> int:
        expansion_params = self.expansion_params if self.expansion_params else self.cfg.model.mdn.mdn_dist_expansion
        if expansion_params is None:
            return 0
        expansion_info = info_resolver(expansion_params)
        n_rbf = expansion_info['n']
        return n_rbf
    
class ProtDimRecorder:
    def __init__(self) -> None:
        self.prot_dim = None
        self.overwrite_count = 0

prot_dim_recorder = ProtDimRecorder()
def get_prot_dim(cfg: Config, want_original=False):
    return 1280

# it is overwritten only when there is a protein embedding transformer layer.
def overwrite_prot_dim(config_dict, val):
    assert "ProtTF" in config_dict["modules"], config_dict["modules"]
    assert prot_dim_recorder.overwrite_count == 0, vars(prot_dim_recorder)
    prot_dim_recorder.prot_dim = val
    prot_dim_recorder.overwrite_count += 1
    

def compute_euclidean_distances_matrix(X, Y, B, N_l):
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    X = X.double()
    Y = Y.double()
    
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    return torch.nan_to_num((dists**0.5),10000)


def to_dense_batch_pth(N, feats, fill_value=0):
    max_num_nodes = int(N.max())
    batch_size = N.shape[0]
    num_total_nodes = N.sum()

    batch = torch.cat([torch.full((1,x.type(torch.int)), y) for x,y in zip(N,range(batch_size))],dim=1).reshape(-1).type(torch.long).to(N.device)
    cum_nodes = torch.cat([batch.new_zeros(1), N.cumsum(dim=0)])
    idx = torch.arange(num_total_nodes, dtype=torch.long, device=N.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, fill_value)
    out[idx] = feats
    out = out.view([batch_size, max_num_nodes] + list(feats.size())[1:])
    
    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                    device=N.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)  
    return out, mask
