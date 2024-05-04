import sys
from typing import List, Union
import os.path as osp
import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean, scatter_min
from Networks.SharedLayers.ActivationFns import activation_getter

from Networks.PairedPropLayers.MDNLayer import MDNLayer, GaussExpandLayer, get_prot_dim
from utils.DataPrepareUtils import pairwise_dist
from utils.LossFn import calculate_probablity
from utils.configs import Config
from utils.data.MyData import MyData
from utils.data.MolFileDataset import SDFDataset
from utils.data.data_utils import get_lig_batch, get_lig_coords, get_num_mols, infer_device, infer_type
from utils.eval.nmdn import NMDN_Calculator
from utils.utils_functions import DistCoeCalculator, lazy_property
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, HeteroData


class MDNPropLayer(MDNLayer, GaussExpandLayer):
    """
    Predict pairwise property using MDN-like MLP layer.
    """
    def __init__(self, mdn_edge_name, cfg: Config, dropout_rate=0.15, n_atom_types=95, martini2aa_pooling=False) -> None:
        MDNLayer.__init__(self, mdn_edge_name, cfg, dropout_rate, n_atom_types, martini2aa_pooling)
        GaussExpandLayer.__init__(self, cfg)

        # not needed
        del self.z_pi, self.z_mu, self.z_sigma

        # cutoff = config_dict["mdn_threshold_prop"]
        # cutoff = None if cutoff == "None" else float(cutoff)
        self.cutoff_needed = cfg["mdn_threshold_prop"]

        self.dist_coe_calculator = DistCoeCalculator(cfg.model.mdn.pair_prop_dist_coe)
        out_dim = self.dist_coe_calculator.coe_dim
        # KL divergence loss inspired by Jocelyn's T5Prop model
        if cfg["loss_metric"] == "kl_div":
            out_dim = 2 * len(cfg["target_names"])
        # readout layer dimension depends on if we have MLP layer or not
        if self.cfg["n_mdnprop_layers"] == 0:
            in_dim: int = self.lig_dim + self.prot_dim + self.n_rbf
            self.readout = nn.Linear(in_dim, out_dim)
        else:
            self.readout = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, runtime_vars: dict):
        h_l, h_p, h_l_i, h_p_j, pl_edge, pair_dist, data_batch, pair_batch = self.unpack_pl_info(runtime_vars)

        embeds = [h_l_i, h_p_j]
        # add distance information (RBF) to each protein-ligand pair
        if self.dist_infuser is not None:
            dist_embed = self.dist_infuser(pair_dist)
            embeds.append(dist_embed)
        pair_embed = torch.cat(embeds, -1)
        pair_embed = self.MLP(pair_embed)
        pair_batch = get_lig_batch(data_batch)[pl_edge[0, :]]
        pair_dist = pair_dist.detach()

        pair_prop: torch.Tensor = self.readout(pair_embed)
        dist_coe: torch.Tensor = self.dist_coe_calculator(pair_dist)
        pair_prop = pair_prop * dist_coe # [-1, 3]
        # pkd_pair = a0 + a1/r + a2/r^2
        pair_prop = pair_prop.sum(dim=-1, keepdim=True) # [-1, 1]
        mol_prop = scatter_add(pair_prop, pair_batch, dim=0, dim_size=runtime_vars["data_batch"].N.shape[0])
        runtime_vars["pair_mol_prop"] = mol_prop
        if self.pair_atom_prop_needed:
            runtime_vars["pair_atom_prop"] = pair_prop
            # the pair embedding is used to computed reference state in NMDN_AuxPropLayer
            runtime_vars["nmdnprop_pair_embed"] = pair_embed
        return runtime_vars
    
    def gather_hidden_dim(self, cfg: dict):
        return cfg["n_mdnprop_hidden"]
    
    def register_mlp_layer(self):
        n_mdnprop_layers = self.cfg["n_mdnprop_layers"]
        if n_mdnprop_layers == 0:
            self.MLP = nn.Identity()
            return

        mlp_layers = [nn.Linear(self.lig_dim + self.prot_dim + self.n_rbf, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)]
        for i in range(n_mdnprop_layers-1):
            mlp_layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.BatchNorm1d(self.hidden_dim), nn.ELU(), nn.Dropout(p=self.dropout_rate)])
        self.MLP = nn.Sequential(*mlp_layers)


class MPNNPairedPropLayer(GaussExpandLayer):
    """
    Predict pairwise property in a Message Passing Neural Network-like fashion.
    "It is just MPNN with extra steps." 
    """
    def __init__(self, cfg: Config, edge_name: str, activation: str, style="add") -> None:
        super().__init__(cfg)
        lig_dim = cfg.model.n_feature

        self.cutoff_needed = cfg.model.mdn.mdn_threshold_prop
        self.style = style
        self.edge_name = edge_name
        prop_dim = self.get_prot_dim(cfg)

        if style == "concat":
            self.G = nn.Linear(self.n_rbf, prop_dim, bias=False)
            self.mlp_neighbor = nn.Sequential(activation_getter(activation), 
                                                nn.Linear(prop_dim, prop_dim), 
                                                activation_getter(activation))
            readout_dim = prop_dim + lig_dim
        else:
            assert style == "add", style
            self.G = nn.Linear(self.n_rbf, lig_dim)
            self.mlp_neighbor = nn.Sequential(activation_getter(activation), 
                                                nn.Linear(prop_dim, lig_dim), 
                                                activation_getter(activation))
            readout_dim = lig_dim
        self.readout_dim = readout_dim
        self.mlp_center = nn.Sequential(activation_getter(activation), 
                                        nn.Linear(lig_dim, lig_dim), 
                                        activation_getter(activation))

        self.dist_coe_calculator = DistCoeCalculator(cfg.model.mdn.pair_prop_dist_coe)
        out_dim = self.dist_coe_calculator.coe_dim

        # The RMSD of ligands after MMFF optimization. Used as a feature to predict pKd.
        self.infuse_rmsd_info: bool = (cfg.data.pre_computed["rmsd_csv"] is not None)
        if self.infuse_rmsd_info:
            self.readout_dim += 1
        if cfg.data.pre_computed["rmsd_expansion"] is not None:
            self.rmsd_rbf_expansion: GaussExpandLayer = GaussExpandLayer(cfg, cfg.data.pre_computed["rmsd_expansion"])
            self.readout_dim += (self.rmsd_rbf_expansion.n_rbf - 1)
        # add physical terms to the final prediction
        self.pkd_phys_norm: float = cfg.model.mdn["pkd_phys_norm"]
        self.pkd_phys_concat: bool = cfg.model.mdn["pkd_phys_concat"]
        self.register_pkd_phys_terms(cfg)
        self.register_readout(out_dim, cfg)

        # It is used during testing: disable this layer to only predict NMDN score
        self.no_pkd_score: bool = cfg.get("no_pkd_score", False)

    def get_prot_dim(self, config_dict: dict):
        return get_prot_dim(config_dict)

    def register_readout(self, out_dim: int, cfg: Config):
        mdn_cfg = cfg.model.mdn
        if mdn_cfg["n_paired_mdn_readout"] == 1:
            self.readout = nn.Sequential(nn.Linear(self.readout_dim, out_dim))
            return
        
        n_hidden: int = mdn_cfg["n_paired_mdn_readout_hidden"]
        layers = [nn.Linear(self.readout_dim, n_hidden), nn.BatchNorm1d(n_hidden), nn.ELU(), nn.Dropout(p=0.15)]
        
        layers.extend((mdn_cfg["n_paired_mdn_readout"] - 2) * 
                      [nn.Linear(n_hidden, n_hidden), 
                       nn.BatchNorm1d(n_hidden), nn.ELU(), 
                       nn.Dropout(p=0.15)])
        layers.append(nn.Linear(n_hidden, out_dim))
        self.readout = nn.Sequential(*layers)

    def register_pkd_phys_terms(self, cfg: Config):
        # pkd_phys_readout layer predicts a (pairwise) coefficient for the physical terms
        # DeltaG, WaterSolv and logP
        # the pairwise coefficients will be summed/averaged to compute the coefficient for the physical terms
        self.pkd_phys_readout = None
        if cfg.model.mdn.pkd_phys_terms is None:
            return
        
        assert isinstance(cfg.model.mdn.pkd_phys_terms, str), cfg.model.mdn.pkd_phys_terms.__class__
        pkd_phys_terms: List[str] = cfg.model.mdn.pkd_phys_terms.split(",")
        self.pkd_phys_terms: List[str] = pkd_phys_terms

        # no need to use specific readout layer for physical terms if we directly concat it
        # to the atom embedding.
        if self.pkd_phys_concat:
            assert self.pkd_phys_norm is not None
            self.readout_dim += len(self.pkd_phys_terms)
            return
        num_phys_terms: int = len(pkd_phys_terms)
        self.pkd_phys_readout = nn.Sequential(nn.Linear(self.readout_dim, num_phys_terms))
        
    def forward(self, runtime_vars: dict):
        if self.no_pkd_score:
            return runtime_vars
        
        # retrieve edge information
        data_batch = runtime_vars["data_batch"]
        pl_edge, pl_dist = self.retrieve_edge_info(data_batch)

        if self.cutoff_needed is not None:
            # only obtain needed pl_edges to avoid un-needed calculation
            pl_dist_mask = (pl_dist <= self.cutoff_needed)
            pl_edge = pl_edge[:, pl_dist_mask]
            pl_dist = pl_dist[pl_dist_mask]
        pl_dist = pl_dist.view(-1, 1)
        pl_expanded_dist = self.dist_infuser(pl_dist)
        h_l_x, h_p_x, pair_batch = self.retrieve_edge_embed(data_batch, runtime_vars, pl_edge)
        
        # forward
        dist_embed = self.G(pl_expanded_dist)
        h_p_x = self.mlp_neighbor(h_p_x)
        neighbor_info = dist_embed * h_p_x
        h_l_x = self.mlp_center(h_l_x)

        if self.style == "concat":
            mpnn_embed = torch.concat([neighbor_info, h_l_x], dim=-1)
        else:
            assert self.style == "add", self.style
            mpnn_embed = neighbor_info + h_l_x
        mpnn_embed = self.modify_embedding(mpnn_embed, data_batch, pair_batch)
        pair_prop = self.readout(mpnn_embed)
        # distance coefficient modifier
        dist_coe: torch.Tensor = self.dist_coe_calculator(pl_dist)
        pair_prop = pair_prop * dist_coe # [-1, 3]
        # pkd_pair = a0 + a1/r + a2/r^2
        pair_prop = pair_prop.sum(dim=-1, keepdim=True) # [-1, 1]
        mol_prop = scatter_add(pair_prop, pair_batch, dim=0, dim_size=get_num_mols(data_batch))
        mol_prop = mol_prop + self.modify_pred(mpnn_embed, data_batch, pair_batch)

        runtime_vars["pair_mol_prop"] = mol_prop
        return runtime_vars
    
    def modify_embedding(self, mpnn_embed: torch.FloatTensor, data_batch: Batch, pair_batch: torch.LongTensor):
        # concat RMSD information into the final embedding
        if self.infuse_rmsd_info:
            # raw rmsd value
            rmsd_batched = data_batch.rmsd.view(-1, 1)[pair_batch, :]
            # expanded rmsd value
            if self.cfg.data.pre_computed.rmsd_expansion is not None:
                rmsd_batched = self.rmsd_rbf_expansion.gaussian_dist_infuser(rmsd_batched)
            mpnn_embed = torch.concat([mpnn_embed, rmsd_batched], dim=-1)
        # concat physical terms into the final embedding
        if self.pkd_phys_concat:
            # [num_mols, 6]
            phys_mol_prop: torch.Tensor = self.retrieve_molprop(data_batch)
            prop_idx: List[int] = [self.idx_mapper[key] for key in self.pkd_phys_terms]
            # [num_mols, num_props]
            phys_mol_prop_wanted = phys_mol_prop[:, prop_idx]
            # [num_pairs, num_props]
            phys_mol_prop_wanted_batched = phys_mol_prop_wanted[pair_batch, :]
            mpnn_embed = torch.concat([mpnn_embed, phys_mol_prop_wanted_batched / self.pkd_phys_norm], dim=-1)
        return mpnn_embed
    
    def modify_pred(self, mpnn_embed: torch.FloatTensor, data_batch: Batch, pair_batch: torch.LongTensor):
        if self.pkd_phys_readout is None:
            return 0.
        
        # integrate the predicted physical terms into the prediction
        # the pairwise interaction terms plus the physical terms.
        # [num_pairs, n_coe]
        pair_pkd_phys_coe: torch.Tensor = self.pkd_phys_readout(mpnn_embed)
        # [num_mols, n_coe]
        mol_pkd_phys_coe = scatter_mean(pair_pkd_phys_coe, pair_batch, dim=0, dim_size=get_num_mols(data_batch))
        # [num_mols, 6]
        phys_mol_prop: torch.Tensor = self.retrieve_molprop(data_batch)
        modify: torch.FloatTensor = 0.
        for i, coe_key in enumerate(self.pkd_phys_terms):
            coe: torch.Tensor = mol_pkd_phys_coe[:, i].view(-1, 1)
            this_phys_mol_prop = phys_mol_prop[:, [self.idx_mapper[coe_key]]]
            modify = modify + coe * this_phys_mol_prop
        return modify
    
    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        # This happens when using ESM-GearNet. In this case, the PL edge is computed on-the-fly.
        if isinstance(data_batch, dict):
            # [num_prot_atoms, 3]
            prot_pos: torch.FloatTensor = data_batch["graph"].node_position
            # [num_lig_atoms, 3]
            lig_pos: torch.FloatTensor = get_lig_coords(data_batch)
            # [num_lig_atoms, num_prot_atoms]
            pair_dist = pairwise_dist(lig_pos, prot_pos)
            # [num_lig_atoms, num_prot_residues]
            pair_dist_res = scatter_min(pair_dist, data_batch["graph"].atom2residue, dim=-1)[0]
            edge_index = torch.nonzero(pair_dist_res<10.)
            pl_dist = pair_dist_res[edge_index[:, 0], edge_index[:, 1]]
            data_batch["esm-gearnet-pl_edge"] = edge_index.T
            data_batch["esm-gearnet-pl_dist"] = pl_dist
            return edge_index.T, pl_dist

        d0 = data_batch.get_example(0) if isinstance(data_batch, Batch) else data_batch
        if isinstance(d0, MyData):
            return getattr(data_batch, f"{self.edge_name}_edge_index"), getattr(data_batch, f"{self.edge_name}_dist")
        
        assert isinstance(d0, HeteroData), data_batch.__class__
        return data_batch[("ligand", "interaction", "protein")].min_dist_edge_index, \
            data_batch[("ligand", "interaction", "protein")].min_dist
    
    def retrieve_edge_embed(self, data_batch: Batch, runtime_vars: dict, edge_index: torch.LongTensor):
        # h_l: ligand embedding by order
        h_l = runtime_vars["vi"]
        # h_p: protein embedding by order
        h_p = runtime_vars["prot_embed"]
        # h_l_x and h_p_x are the embedding based on edges
        h_l_x = h_l[edge_index[0, :], :]
        h_p_x = h_p[edge_index[1, :], :]

        pair_batch = get_lig_batch(data_batch)[edge_index[0, :]]
        return h_l_x, h_p_x, pair_batch
    
    def retrieve_molprop(self, data_batch: Batch):
        from utils.LossFn import post_calculation
        # [num_mols, 1]
        if hasattr(data_batch, "mol_prop"):
            # somehow I saved the mol properties as np.ndarray, so I have to convert them back to tensors.
            mol_prop = data_batch.mol_prop
            if isinstance(mol_prop, list):
                mol_prop = torch.concat([torch.as_tensor(t).view(1, -1) for t in mol_prop], dim=0)
            mol_prop = mol_prop.view(get_num_mols(data_batch), -1)
            if mol_prop.shape[1] == 3:
                mol_prop: torch.Tensor = post_calculation(mol_prop)
            return mol_prop.to(infer_device(data_batch))
        
        if hasattr(data_batch, "atom_prop"):
            # [n_mols, 6]
            atom_prop: torch.Tensor = post_calculation(data_batch.atom_prop)
            phys_atom_prop = atom_prop
            phys_mol_prop: torch.Tensor = scatter_add(phys_atom_prop, get_lig_batch(data_batch), dim=0, dim_size=get_num_mols(data_batch))
            return phys_mol_prop
        
        # atom_prop can be computed on the fly during evaluation
        assert not self.training, "atom_prop must be pre-computed during training"
        exp_id: int = int(osp.basename(self.cfg["folder_prefix"]).split("_")[-1])
        # the atomprop is computed in a different way after experiment 469.
        # I keep the original script for reproducibility
        if exp_id <= 469 or (exp_id >= 484 and exp_id <= 485):
            # previously I used data_batch to directly compute the atomprop
            # but it only has polar hydrogens, which may cause problems
            # After experiment 484, I processed the data set with all H so it should be fine.
            model_pred = self.atomprop_predictor.predict_nograd(data_batch)
            phys_atom_prop = model_pred["atom_prop"]
            phys_mol_prop: torch.Tensor = scatter_add(phys_atom_prop, get_lig_batch(data_batch), dim=0, dim_size=get_num_mols(data_batch))
            return post_calculation(phys_mol_prop)
        
        if self.cfg["short_name"] == "casf2016-scoring":
            pdb_list: List[str] = data_batch.pdb
            allh_sdf_files: List[str] = [self.casf_reader.pdb2lig_core_sdf(pdb) for pdb in pdb_list]
            sdf_ds = SDFDataset(allh_sdf_files)
            dl = DataLoader(sdf_ds, batch_size=len(allh_sdf_files))
            allh_batch = next(iter(dl))
            model_pred = self.atomprop_predictor.predict_nograd(allh_batch)
            phys_atom_prop = model_pred["atom_prop"]
            phys_mol_prop: torch.Tensor = scatter_add(phys_atom_prop, allh_batch.atom_mol_batch, dim=0, dim_size=get_num_mols(data_batch))
            phys_mol_prop = post_calculation(phys_mol_prop)
            return phys_mol_prop
        if self.cfg["short_name"] == "casf2016-screening" and exp_id in [489, 530, 531]:
            # fake mol prop, temp solution
            phys_mol_prop = torch.ones(get_num_mols(data_batch), 6).to(infer_device(data_batch)).type(infer_type(data_batch))
            return phys_mol_prop
        raise NotImplementedError
    
    @lazy_property
    def idx_mapper(self):
        return {"gasEnergy": 0, "watEnergy": 1, "octEnergy": 2,
                "watSol": 3, "octSol": 4, "watOct": 5}
    
    @lazy_property
    def atomprop_predictor(self):
        from utils.eval.predict import EnsPredictor
        return EnsPredictor("./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*")
    
    @lazy_property
    def casf_reader(self):
        from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
        return CASF2016Reader("/CASF-2016-cyang")


class NMDN_AuxPropLayer(MDNPropLayer):
    """
    Compute pKd using NMDN score:
    pKd_ij = self.MLP(h_l_i, h_p_j) * NMDN_ij
    """
    def __init__(self, mdn_edge_name, config_dict, dropout_rate=0.15, n_atom_types=95, martini2aa_pooling=False) -> None:
        super().__init__(mdn_edge_name, config_dict, dropout_rate, n_atom_types, martini2aa_pooling)
        self._sanity_checked: bool = False
        self.nmdn_calculator = NMDN_Calculator(config_dict, [config_dict["auxprop_nmdn_name"]])
        self.pair_atom_prop_needed = True
        self.auxprop_nmdn_compute_ref: bool = config_dict["auxprop_nmdn_compute_ref"]
        if self.auxprop_nmdn_compute_ref:
            self.readout_ref = nn.Linear(self.hidden_dim + 1, 1)

    def forward(self, runtime_vars: dict):
        h_l, h_p, h_l_i, h_p_j, pl_edge, pair_dist, data_batch, pair_batch = self.unpack_pl_info(runtime_vars)
        runtime_vars: dict = super().forward(runtime_vars)
        self.check_sanity(runtime_vars)
        pi, sigma, mu, dist = runtime_vars["pi"], runtime_vars["sigma"], runtime_vars["mu"], runtime_vars["dist"]
        pair_prob = calculate_probablity(pi, sigma, mu, dist)
        nmdn_out: dict = self.nmdn_calculator.compute_pair_lvl_scores(pair_prob, runtime_vars, data_batch)

        nmdn_score: Tensor = nmdn_out[self.cfg["auxprop_nmdn_name"]].view(-1, 1)
        if self.auxprop_nmdn_compute_ref:
            pair_embed = torch.concat([runtime_vars["nmdnprop_pair_embed"], nmdn_score], dim=-1)
            ref = self.readout_ref(pair_embed)
            nmdn_score = nmdn_score - ref
        pair_coe: Tensor = runtime_vars["pair_atom_prop"].view(-1, 1)
        mdn_aux_score: Tensor = nmdn_score * pair_coe
        pair_batch = get_lig_batch(data_batch)[pl_edge[0, :]]
        mol_prop = scatter_add(mdn_aux_score, pair_batch, dim=0, dim_size=get_num_mols(data_batch))
        runtime_vars["pair_mol_prop"] = mol_prop
        del runtime_vars["pair_atom_prop"]

        return runtime_vars
    
    def check_sanity(self, runtime_vars: dict):
        # making sure all needed properties are available.
        if self._sanity_checked: return

        for key in ["pi", "sigma", "mu", "dist", "C_batch"]:
            assert key in runtime_vars, f"{key} is not in runtime vars, make sure MDNLayer is called before MDN_AuxPropLayer"
        assert "pair_atom_prop" in runtime_vars, "pair_atom_prop not properly predicted, make sure self.pair_atom_prop_needed == True"
        self._sanity_checked = True
