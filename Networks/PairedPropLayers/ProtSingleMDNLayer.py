from torch_scatter import scatter_mean
from Networks.PairedPropLayers.MDNLayer import get_prot_dim
from Networks.PairedPropLayers.NotEnoughMDNLayers import GeneralMDNLayer, ProtProtMDNLayer


import torch
from torch import Tensor
from torch_geometric.data import Batch, Data


from typing import Dict, List, Optional, Tuple, Union

from utils.LossFn import mdn_loss_fn


class ProtSingleMDNLayer(GeneralMDNLayer):
    # Add protein intra-residue interaction.
    def __init__(self, ltype="non-local", **kwargs) -> None:
        super().__init__("prot_embed", "prot_embed", **kwargs)
        # exlcude close interactions. For example, protprot_exclude_edge==1 means exlcude 1-2 interaction
        # protprot_exclude_edge==2 means exclude 1-2 and 1-3 interaction.
        self.seq_separation_cutoff: Optional[int] = kwargs["cfg"]["protprot_exclude_edge"]
        self.ltype = ltype

    def overwrite_lig_dim(self) -> Optional[int]:
        return get_prot_dim(self.cfg)

    def unpack_pl_info(self, runtime_vars: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Data]:
        h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, pair_batch = super().unpack_pl_info(runtime_vars)

        # exlcude close interactions. For example, protprot_exclude_edge==1 means exlcude 1-2 interaction
        # protprot_exclude_edge==2 means exclude 1-2 and 1-3 interaction.
        if self.seq_separation_cutoff is not None:
            edge_dist: torch.LongTensor = (pp_edge[0] - pp_edge[1]).abs()
            if self.ltype == "non-local":
                edge_mask: torch.BoolTensor = edge_dist > self.seq_separation_cutoff
            else:
                assert self.ltype == "local"
                edge_mask: torch.BoolTensor = edge_dist <= self.seq_separation_cutoff
            pp_edge = pp_edge[:, edge_mask]
            pp_dist = pp_dist[edge_mask]
            h_1_i = h_1_i[edge_mask, :]
            h_2_j = h_2_j[edge_mask, :]
            pair_batch = pair_batch[edge_mask]
        # only obtain needed pl_edges to avoid un-needed calculation
        if self.cutoff_needed is not None:
            this_pl_dist_mask = (pp_dist <= self.cutoff_needed).view(-1)
            pp_edge = pp_edge[:, this_pl_dist_mask]
            pp_dist = pp_dist[this_pl_dist_mask]

        return h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, pair_batch
    
    def retrieve_edge_info(self, data_batch: Union[Batch, dict]):
        bond_type = f"PP_min_dist"
        pp_dist = getattr(data_batch, f"{bond_type}_oneway_dist")
        edge_index = getattr(data_batch, f"{bond_type}_oneway_edge_index")
        return edge_index, pp_dist
    
