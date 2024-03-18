from torch_scatter import scatter_mean
from Networks.PairedPropLayers.NotEnoughMDNLayers import ProtProtMDNLayer


import torch
from torch import Tensor
from torch_geometric.data import Data


from typing import Dict, List, Optional, Tuple

from utils.LossFn import mdn_loss_fn


class ProtProtIntraMDNLayer(ProtProtMDNLayer):
    # Add protein intra-residue interaction.
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hist_pp_intra_mdn: bool = kwargs["cfg"]["hist_pp_intra_mdn"]
        # exlcude close interactions. For example, protprot_exclude_edge==1 means exlcude 1-2 interaction
        # protprot_exclude_edge==2 means exclude 1-2 and 1-3 interaction.
        self.protprot_exclude_edge: Optional[int] = kwargs["cfg"]["protprot_exclude_edge"]
        # record 1-2, 1-3, 1-4, ... interaction information
        self._res_num_diff: List[torch.LongTensor] = None

    def forward_ext_prot_embed(self, runtime_vars: dict):
        self._res_num_diff = []
        res = super().forward_ext_prot_embed(runtime_vars)
        res.update(self.compute_pp_intra_hist(res))
        self._res_num_diff = None
        return res

    def unpack_pl_info(self, runtime_vars: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Data]:
        h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, pair_batch = super().unpack_pl_info(runtime_vars)
        if not self.training and self.hist_pp_intra_mdn:
            # inter protein interactions are recorded as -1 and will be removed later on.
            self._res_num_diff.append(torch.zeros_like(pp_dist).long().fill_(-1).view(-1))
        n_inter_edges: int = pp_edge.shape[-1]
        # According to experience, the number of intra- interactions is way more than inter interactions
        # So we only randomly sample some edges during training to reduce computational cost.
        if self.training:
            sel_idx: torch.LongTensor = torch.randperm(n_inter_edges, device=h_1.get_device())[:n_inter_edges//3]
            pp_edge, pp_dist = pp_edge[:, sel_idx], pp_dist[sel_idx]
            h_1_i, h_2_j = h_1_i[sel_idx, :], h_2_j[sel_idx, :]
        h_1_i, h_2_j, pp_edge, pp_dist, pair_batch = [h_1_i], [h_2_j], [pp_edge], [pp_dist], [pair_batch]
        hi = {"h_1": h_1, "h_2": h_2}
        for chain_num in [1, 2]:
            # intra-residue interaction indices
            this_pp_edge: torch.LongTensor = getattr(data_batch, f"PP_min_dist_chain{chain_num}_oneway_edge_index")
            this_pp_dist: torch.DoubleTensor = getattr(data_batch, f"PP_min_dist_chain{chain_num}_oneway_dist")
            # exlcude close interactions. For example, protprot_exclude_edge==1 means exlcude 1-2 interaction
            # protprot_exclude_edge==2 means exclude 1-2 and 1-3 interaction.
            if self.protprot_exclude_edge is not None:
                edge_dist: torch.LongTensor = (this_pp_edge[0] - this_pp_edge[1]).abs()
                edge_mask: torch.BoolTensor = edge_dist > self.protprot_exclude_edge
                this_pp_edge = this_pp_edge[:, edge_mask]
                this_pp_dist = this_pp_dist[edge_mask]
            # only obtain needed pl_edges to avoid un-needed calculation
            if self.cutoff_needed is not None:
                this_pl_dist_mask = (this_pp_dist <= self.cutoff_needed)
                this_pp_edge = this_pp_edge[:, this_pl_dist_mask]
                this_pp_dist = this_pp_dist[this_pl_dist_mask]
            # According to experience, the number of intra- interactions is way more than inter interactions
            # So we only randomly sample some edges during training to reduce computational cost.
            if self.training:
                n_intra_edges: int = this_pp_edge.shape[-1]
                sel_idx: torch.LongTensor = torch.randperm(n_intra_edges, device=h_1.get_device())[:n_inter_edges//3]
                this_pp_edge, this_pp_dist = this_pp_edge[:, sel_idx], this_pp_dist[sel_idx]
            h_1_i.append(hi[f"h_{chain_num}"][this_pp_edge[0]])
            h_2_j.append(hi[f"h_{chain_num}"][this_pp_edge[1]])
            pp_edge.append(this_pp_edge)
            pp_dist.append(this_pp_dist.view(-1, 1))
            # pair_batch: mapping the pair to the i_th molecule in the batch
            pair_batch.append(getattr(data_batch, f"atom_mol_batch_chain{chain_num}")[this_pp_edge[0]])
            if not self.training and self.hist_pp_intra_mdn:
                self._res_num_diff.append((this_pp_edge[0] - this_pp_edge[1]).abs())
        h_1_i, h_2_j = torch.concat(h_1_i), torch.concat(h_2_j)
        pp_edge = torch.concat(pp_edge, dim=-1)
        pp_dist = torch.concat(pp_dist, dim=0)
        pair_batch = torch.concat(pair_batch)
        return h_1, h_2, h_1_i, h_2_j, pp_edge, pp_dist, data_batch, pair_batch
    
    def compute_pp_intra_hist(self, model_output: Dict[str, torch.Tensor]):
        if not self.hist_pp_intra_mdn: return {}
        if self.training: return {}

        res_num_diff: torch.LongTensor = torch.concat(self._res_num_diff, dim=0)
        mdn: torch.Tensor = mdn_loss_fn(model_output["pi"], model_output["sigma"], model_output["mu"], model_output["dist"])

        assert mdn.shape[0] == res_num_diff.shape[0], f"{mdn.shape} vs. {res_num_diff.shape}"

        intra_mask: torch.BoolTensor = (res_num_diff >= 0)
        mdn_hist: torch.Tensor = scatter_mean(mdn[intra_mask], res_num_diff[intra_mask], dim=0)
        return {"mdn_hist": mdn_hist}
