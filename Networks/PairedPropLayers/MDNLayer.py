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
from utils.data.MyData import MyData
from utils.data.data_utils import get_lig_batch, get_lig_z

from utils.utils_functions import DistCoeCalculator, gaussian_rbf, get_device, info_resolver, lazy_property, softplus_inverse, floating_type

class MDNLayer(nn.Module):
    """
    A Mixture Density Network to learn the probability density distribution of the distance between
    ligand atom and protein (Martini bead or residues).
    """
    def __init__(self, mdn_edge_name, cfg: dict, dropout_rate=0.15, n_atom_types=95, martini2aa_pooling=False) -> None:
        super().__init__()
        self.lig_atom_types: bool = cfg["mdn_w_lig_atom_types"] > 0.
        self.prot_atom_types: bool = cfg["mdn_w_prot_atom_types"] > 0.
        self.lig_atom_props: bool = cfg["mdn_w_lig_atom_props"] > 0.
        self.prot_sasa: bool = cfg["mdn_w_prot_sasa"] > 0.
        self.dropout_rate = dropout_rate
        self.config_dict = cfg

        self.prot_dim = self.get_prot_dim()
        self.lig_dim = cfg["n_feature"]
        if "KANO" in cfg["modules"].split(" "):
            # KANO embed atoms into 300
            self.lig_dim = 300
        if "Comb-P-KANO" in cfg["modules"].split(" "):
            # KANO embed atoms into 300
            assert "KANO" in cfg["modules"].split(" "), cfg["modules"]
            self.lig_dim = 300 + cfg["n_feature"]
        if self.overwrite_lig_dim(): self.lig_dim = self.overwrite_lig_dim()

        # use external embedding for proteins. For example, ESM embeddings
        self.ext_prot_embed = (cfg["prot_embedding_root"] is not None) or \
            ("ESMGearnet" in cfg["modules"].split())
        if self.ext_prot_embed:
            self.cutoff_needed = max(cfg["mdn_threshold_train"], cfg["mdn_threshold_eval"])

        # used to calculate cross MDN pKd loss
        self.pair_atom_prop_needed = (cfg["w_cross_mdn_pkd"] > 0.)
        self.cross_mdn_prop_name = cfg["cross_mdn_prop_name"]
        if self.pair_atom_prop_needed:
            # the cutoff should be the same as prop cutoff to calculate pair-wise loss
            self.cutoff4cross_loss = cfg["mdn_threshold_prop"]
            self.prob_transform = nn.Linear(1, 1)
            self.nll_transform = nn.Linear(1, 1)

        hidden_dim = self.gather_hidden_dim(cfg)
        assert hidden_dim is not None
        self.hidden_dim = hidden_dim
        self.register_mlp_layer()
        n_gaussians=cfg["n_mdn_gauss"]
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

        self.voronoi_edge = cfg["mdn_voronoi_edge"]

        # Only enabled when there is a martini2aa pooling layer before the MDN layer. 
        # In this case, we should pull out AA positions for AA-level MDN calculation.
        self.martini2aa_pooling = martini2aa_pooling
        # the action to the embedding after pooling
        self.martini2aa_action = cfg["martini2aa_action"]

        self.mdn_edge_name = mdn_edge_name
        if self.mdn_edge_name == "None":
            self.mdn_edge_name = "PL_oneway"

    def get_prot_dim(self):
        return get_prot_dim(self.config_dict)

    def gather_hidden_dim(self, cfg: dict):
        return cfg["n_mdn_hidden"]

    def register_mlp_layer(self):
        mlp_layers = [nn.Linear(self.lig_dim + self.prot_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)]
        for i in range(self.config_dict["n_mdn_layers"]-1):
            mlp_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)])
        self.MLP = nn.Sequential(*mlp_layers)

    def forward(self, runtime_vars: dict):
        if self.martini2aa_pooling:
            return self.forward_old(runtime_vars)
        elif self.ext_prot_embed:
            return self.forward_ext_prot_embed(runtime_vars)

        data_batch = runtime_vars["data_batch"]
        h = runtime_vars["vi"]
        pl_edge = getattr(data_batch, f"{self.mdn_edge_name}_edge_index")
        pl_dist = getattr(data_batch, f"{self.mdn_edge_name}_dist")

        h_l_x = h[pl_edge[1, :]]
        h_p_x = h[pl_edge[0, :]]

        C = torch.cat((h_l_x, h_p_x), -1)
        C = self.MLP(C)
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        out = {}
        out["pi"] = pi
        out["sigma"] = sigma
        out["mu"] = mu
        out["dist"] = pl_dist.unsqueeze(1).detach()
        out["C_batch"] = get_lig_batch(data_batch)[pl_edge[1, :]]

        mol2lig_mask = (data_batch.mol_type == 1)
        mol2prot_mask = (data_batch.mol_type == 0)
        h_l = h[mol2lig_mask, :]
        h_p = h[mol2prot_mask, :]
        if self.lig_atom_types:
            lig_atom_types = self.lig_atom_types_layer(h_l)
            lig_atom_types_label = data_batch.Z[mol2lig_mask]
            out["lig_atom_types"] = lig_atom_types
            out["lig_atom_types_label"] = lig_atom_types_label
        if self.prot_atom_types:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_atom_types = self.prot_atom_types_layer(h_p)
            prot_atom_types_label = data_batch.Z[mol2prot_mask]
            out["prot_atom_types"] = prot_atom_types
            out["prot_atom_types_label"] = prot_atom_types_label
        if self.lig_atom_props:
            lig_atom_props = self.lig_atom_props_layer(h_l)
            out["lig_atom_props"] = lig_atom_props
        if self.prot_sasa:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_sasa = self.prot_sasa_layer(h_p)
            out["prot_sasa"] = prot_sasa
        return out
    
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

        out = self.update_pair_prop(out, pi, sigma, mu, pl_dist)
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
    
    def update_pair_prop(self, out: dict, pi, sigma, mu, pl_dist):
        if not self.pair_atom_prop_needed:
            return out
        # Regularization terms to improve the scoring power of MDN score
        # does not work as expected, therefore discarded

        if self.cross_mdn_prop_name == "pair_prob_transformed":
            pair_prob = calculate_probablity(pi, sigma, mu, pl_dist)
            pl_dist_mask = (pl_dist <= self.cutoff4cross_loss).view(-1)
            pair_prob = pair_prob[pl_dist_mask].view(-1, 1)
            pair_prob_transformed = self.prob_transform(pair_prob)
            out[self.cross_mdn_prop_name] = pair_prob_transformed
        else:
            assert self.cross_mdn_prop_name == "pair_nll_transformed", self.cross_mdn_prop_name
            pair_nll = mdn_loss_fn(pi, sigma, mu, pl_dist)
            pl_dist_mask = (pl_dist <= self.cutoff4cross_loss).view(-1)
            pair_nll = pair_nll[pl_dist_mask].view(-1, 1)
            pair_nll_transformed = self.nll_transform(pair_nll)
            out[self.cross_mdn_prop_name] = pair_nll_transformed
        return out

    def forward_old(self, module_out_raw):
        # a less efficient way to compute the MDN loss: all PL pairs are computed and then masked out based on distance.
        # it will cause PyTorch to compute a lot of unessecary gradients which consumes a lot of memory and computational time.
        data_batch = module_out_raw["data_batch"]

        if "h_l" in module_out_raw.keys():
            assert self.martini2aa_pooling
            h_l = module_out_raw["h_l"]
            h_p = module_out_raw["h_p"]
        else:
            h = module_out_raw["vi"]

        if self.voronoi_edge:
            assert not self.martini2aa_pooling
            return self.forward_voronoi(h, data_batch)

        mol2lig_mask = (data_batch.mol_type == 1)
        mol2prot_mask = (data_batch.mol_type == 0)
        if self.martini2aa_pooling:
            # assuming h_l and h_p are provided by the previous pooling layer
            # concat h_p with pre-calculated features (inter-residue distances and dihediral angles)
            # here we discard 9 dimensions from h_p to make sure the dimension keeps the same. Hope it works.
            if self.martini2aa_action == "replace_with_aa_feats":
                h_p = torch.concat([h_p[:, :-9], data_batch.feats_prot_aa], dim=-1)
            else:
                assert self.martini2aa_action == "ignore", self.martini2aa_action
            assert h_l is not None
            h_p_x, p_mask = to_dense_batch_pth(data_batch.N_p_aa, h_p)
        else:
            h_l = h[mol2lig_mask, :]
            h_p = h[mol2prot_mask, :]
            h_p_x, p_mask = to_dense_batch_pth(data_batch.N_p, h_p)
        h_l_x, l_mask = to_dense_batch_pth(data_batch.N_l, h_l)

        h_l_pos, __ = to_dense_batch_pth(data_batch.N_l, data_batch.R[mol2lig_mask, :])
        if self.martini2aa_pooling:
            h_p_pos, __ = to_dense_batch_pth(data_batch.N_p_aa, data_batch.R_aa)
        else:
            h_p_pos, __ = to_dense_batch_pth(data_batch.N_p, data_batch.R[mol2prot_mask, :])

        # B: batch_size, N_l: number of atoms in ligand, N_p: number of Martini Beads in protein
        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)

        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1) # [B, N_l, N_p, C_out]
        
        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1) # [B, N_l, N_p, C_out]

        C = torch.cat((h_l_x, h_p_x), -1)
        C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        C = C[C_mask]
        C = self.MLP(C)
        
        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]

        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        
        dist = compute_euclidean_distances_matrix(h_l_pos, h_p_pos.view(B,-1,3), B, N_l)[C_mask]
        out = {}
        out["pi"] = pi
        out["sigma"] = sigma
        out["mu"] = mu
        out["dist"] = dist.unsqueeze(1).detach()
        out["C_batch"] = C_batch

        if self.lig_atom_types:
            lig_atom_types = self.lig_atom_types_layer(h_l)
            lig_atom_types_label = data_batch.Z[mol2lig_mask]
            out["lig_atom_types"] = lig_atom_types
            out["lig_atom_types_label"] = lig_atom_types_label
        if self.prot_atom_types:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_atom_types = self.prot_atom_types_layer(h_p)
            prot_atom_types_label = data_batch.Z[mol2prot_mask]
            out["prot_atom_types"] = prot_atom_types
            out["prot_atom_types_label"] = prot_atom_types_label
        if self.lig_atom_props:
            lig_atom_props = self.lig_atom_props_layer(h_l)
            out["lig_atom_props"] = lig_atom_props
        if self.prot_sasa:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_sasa = self.prot_sasa_layer(h_p)
            out["prot_sasa"] = prot_sasa
        return out

    def forward_voronoi(self, h, data_batch):
        """
        Instead of using distance cutoff for MDN loss calculation, we use voronoi edge.
        """
        pl_voronoi_edge = data_batch.PL_Voronoi1_edge_index

        h_l_x = h[pl_voronoi_edge[0, :]]
        h_p_x = h[pl_voronoi_edge[1, :]]

        l_pos = data_batch.R[pl_voronoi_edge[0, :], :]
        p_pos = data_batch.R[pl_voronoi_edge[1, :], :]
        dist = torch.sqrt(torch.sum((l_pos - p_pos)**2, dim=-1))

        C = torch.cat((h_l_x, h_p_x), -1)
        C = self.MLP(C)
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        out = {}
        out["pi"] = pi
        out["sigma"] = sigma
        out["mu"] = mu
        out["dist"] = dist.unsqueeze(1).detach()
        out["C_batch"] = get_lig_batch(data_batch)[pl_voronoi_edge[0, :]]

        if self.lig_atom_types:
            mol2lig_mask = (data_batch.mol_type == 1)
            h_l = h[mol2lig_mask, :]
            lig_atom_types = self.lig_atom_types_layer(h_l)
            lig_atom_types_label = data_batch.Z[mol2lig_mask]
            out["lig_atom_types"] = lig_atom_types
            out["lig_atom_types_label"] = lig_atom_types_label
        if self.prot_atom_types:
            mol2prot_mask = (data_batch.mol_type == 0)
            h_p = h[mol2prot_mask, :]
            prot_atom_types = self.prot_atom_types_layer(h_p)
            prot_atom_types_label = data_batch.Z[mol2prot_mask]
            out["prot_atom_types"] = prot_atom_types
            out["prot_atom_types_label"] = prot_atom_types_label
        return out
    
    def overwrite_lig_dim(self) -> Optional[int]:
        return None
    
class GaussExpandLayer(nn.Module):
    def __init__(self, config_dict, expansion_params: str = None) -> None:
        # avoid calling nn.Module twice when which seems to reset all parameters in the module
        # it is nessesary in the double inheritance in the class MDNPropLayer
        if '_buffers' not in self.__dict__:
            super().__init__()
        self.expansion_params: str = expansion_params
        self.config_dict = config_dict
        self.register_expansion_params()

    def gaussian_dist_infuser(self, pair_dist: torch.Tensor):
        rbf = gaussian_rbf(pair_dist, self.centers, self.widths, self.cutoff, self.expansion_coe, self.linear_gaussian)
        return rbf
    
    def register_expansion_params(self):
        # register expansion params
        expansion_params = self.expansion_params if self.expansion_params else self.config_dict["mdn_dist_expansion"]
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
        expansion_params: str =  self.expansion_params if self.expansion_params else self.config_dict["mdn_dist_expansion"]
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
def get_prot_dim(config_dict, want_original=False):
    # ESM-Gearnet protein embedding.
    if "ESMGearnet" in config_dict["modules"].split():
        return 4352 
    # if we do not use pre-computed protein embedding, protein dimension is the same as n_feature
    if (config_dict["prot_embedding_root"] is None):
        return config_dict["n_feature"]
    # assert we are accessing the un-transformed protein embedding
    if want_original:
        assert prot_dim_recorder.overwrite_count == 0, vars(prot_dim_recorder)
    
    # get external protein dimension in a lazy way. i.e., only run when needed.
    if prot_dim_recorder.prot_dim is not None:
        return prot_dim_recorder.prot_dim
    
    example_pth = glob(osp.join(config_dict["prot_embedding_root"].split(";")[0], "*.pth"))[1]
    example_d = torch.load(example_pth, map_location=get_device())
    if isinstance(example_d, dict):
        key = list(example_d.keys())[0]
        example_d = example_d[key]
    prot_dim_recorder.prot_dim = example_d.shape[-1]
    return prot_dim_recorder.prot_dim

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
