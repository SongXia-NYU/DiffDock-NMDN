from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from Networks.PairedPropLayers.MDNLayer import MDNLayer, get_prot_dim
import torch.nn.functional as F

from utils.configs import Config
from utils.data.data_utils import get_lig_batch


class GeneralMDNLayer(MDNLayer):
    def __init__(self, h1_name: str, h2_name: str, mdn_edge_name: str, **kwargs) -> None:
        self.h1_name = h1_name
        self.h2_name = h2_name
        super().__init__(mdn_edge_name=mdn_edge_name, **kwargs)

    def unpack_pl_info(self, runtime_vars: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Data, Tensor]:
        data_batch = runtime_vars["data_batch"]
        # h_1: chain 1 embedding 
        h_1 = runtime_vars[self.h1_name]
        # h_2: chain 2 embedding 
        h_2 = runtime_vars[self.h2_name]

        edge_index, dist = self.retrieve_edge_info(data_batch)

        if self.cutoff_needed is not None:
            # only obtain needed pl_edges to avoid un-needed calculation
            dist_mask = (dist <= self.cutoff_needed)
            edge_index = edge_index[:, dist_mask]
            dist = dist[dist_mask]
        dist = dist.view(-1, 1)
        # h_l_x and h_p_x are the embedding based on edges
        h_1_i = h_1[edge_index[0, :], :]
        h_2_j = h_2[edge_index[1, :], :]
        pair_batch = get_lig_batch(data_batch)[edge_index[0, :]]
        return h_1, h_2, h_1_i, h_2_j, edge_index, dist, data_batch, pair_batch


    def forward_ext_prot_embed(self, runtime_vars: dict):
        h_1, h_2, h_1_i, h_2_j, edge_index, dist, data_batch, pair_batch = self.unpack_pl_info(runtime_vars)

        embeds = [h_1_i, h_2_j]
        pair_embed = torch.cat(embeds, -1)
        pair_embed = self.MLP(pair_embed)

        # Outputs
        pi, sigma, mu = self.predict_gaussian_params(pair_embed)
        dist = dist.detach()
        if "pi" in runtime_vars:
            # During heterogenous NMDN model, the interaction between metal and ligand is considered by different NMDN layers
            pi, sigma, mu = torch.concat([runtime_vars["pi"], pi]), torch.concat([runtime_vars["sigma"], sigma]), \
                            torch.concat([runtime_vars["mu"], mu])
            dist = torch.concat([runtime_vars["pp_dist"], dist])
            pair_batch = torch.concat([runtime_vars["C_batch"], pair_batch])
            edge_index = torch.concat([runtime_vars["pl_edge_index_used"], edge_index], dim=-1)
        out = {"pi": pi, "sigma": sigma, "mu": mu, "dist": dist,
               "C_batch": pair_batch, "pl_edge_index_used": edge_index}

        assert not self.prot_atom_types
        if self.lig_atom_types:
            lig_atom_types = self.lig_atom_types_layer(h_1)
            lig_atom_types_label = data_batch.Z
            out["lig_atom_types"] = lig_atom_types
            out["lig_atom_types_label"] = lig_atom_types_label
        # these aux tasks are not applicable
        assert not self.prot_atom_types
        assert not self.lig_atom_props
        if self.prot_sasa:
            # Save the problem for the future
            assert not self.martini2aa_pooling
            prot_sasa = self.prot_sasa_layer(h_2)
            out["prot_sasa"] = prot_sasa
            
        runtime_vars.update(out)
        return runtime_vars
    

class ProtProtMDNLayer(GeneralMDNLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__("prot_embed_chain1", "prot_embed_chain2", **kwargs)
    def overwrite_lig_dim(self) -> Optional[int]:
        return get_prot_dim(self.cfg)
    
class KanoProtMDNLayer(GeneralMDNLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__("kano_atom_embed", "prot_embed", **kwargs)

class ComENetProtMDNLayer(GeneralMDNLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__("comenet_atom_embed", "prot_embed", **kwargs)

class MetalLigMDNLayer(GeneralMDNLayer):
    def __init__(self, **kwargs) -> None:
        super().__init__("vi", "kano_atom_embed", **kwargs)

    def forward(self, runtime_vars: dict):
        if runtime_vars["kano_atom_embed"] is None:
            return runtime_vars
        return super().forward(runtime_vars)

    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        pl_edge = data_batch[("ligand", "interaction", "ion")].min_dist_edge_index
        pl_dist = data_batch[("ligand", "interaction", "ion")].min_dist
        return pl_edge, pl_dist
    
    def get_prot_dim(self):
        # the "prot_dim" is the dimension of the ion metal embedding here
        # since ion is encoded by KANO-Metal, the dimension is 300
        return 300
    
    def gather_hidden_dim(self, cfg: Config):
        if cfg.model.mdn["n_mdn_lig_metal_hidden"] is not None:
            return cfg.model.mdn["n_mdn_lig_metal_hidden"]
        return super().gather_hidden_dim(cfg)
