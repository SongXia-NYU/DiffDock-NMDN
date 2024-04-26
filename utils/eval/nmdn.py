from typing import Dict, List, Optional, Set
from torch import Tensor, BoolTensor
import copy

import torch
import numpy as np
from torch_scatter import scatter_add, scatter_max
from torch.distributions import Normal
from torch_geometric.data import Data

from utils.configs import Config
from utils.data.data_utils import get_num_mols, infer_device


class NMDN_Calculator:
    def __init__(self, cfg: Config, requested_score_names: List[str] = None) -> None:
        mdn_cfg = cfg.model.mdn
        self.mdn_threshold_train = mdn_cfg["mdn_threshold_train"]
        self.mdn_threshold_eval = mdn_cfg["mdn_threshold_eval"]

        self.requested_score_names: Set[str] = requested_score_names

    def __call__(self, pair_prob: Optional[torch.Tensor], model_output: dict, data_batch: Data) -> Dict[str, Tensor]:
        """
        Computing external MDN scores using references.
        """
        if pair_prob is None:
            pair_prob = calculate_probablity(model_output["pi"], model_output["sigma"], model_output["mu"], model_output["dist"])
        # pair_batch maps pair_id to mol_id
        pair_batch = model_output["C_batch"].to(infer_device(data_batch))
        n_mols = get_num_mols(data_batch)

        pair_lvl_scores: Dict[str, Tensor] = self.compute_pair_lvl_scores(pair_prob, model_output, data_batch)
        mol_lvl_scores: Dict[str, Tensor] = {}
        for key in pair_lvl_scores:
            mol_lvl_scores[key] = scatter_add(pair_lvl_scores[key], pair_batch, dim=0, dim_size=n_mols)
        return mol_lvl_scores
    
    def compute_pair_lvl_scores(self, pair_prob: torch.Tensor, model_output: dict, data_batch: Data) -> Dict[str, Tensor]:
        out_dict = {}
        # only pairwise probability within a threshold is used for evaluation
        cutoff_mask: BoolTensor = (model_output["dist"] > self.mdn_threshold_eval).view(-1)

        # pair-wise probability
        pair_prob_zeroed = torch.where(cutoff_mask, torch.zeros_like(pair_prob), pair_prob)
        # the square of distance for normalization purposes
        dist_sq: torch.Tensor = (model_output["dist"] ** 2).view(-1)
        if self.to_compute("MDN_SUM_DIST2"):
            sum_dist2_prob: torch.Tensor = pair_prob_zeroed/dist_sq
            out_dict["MDN_SUM_DIST2"] = sum_dist2_prob
        # assuming we have unified the thresholds
        assert self.mdn_threshold_eval == self.mdn_threshold_train
        # compute reference probablity by averaging values between [cutoff-0.5 and cutoff]
        pair_prob_ref_mean = []
        pair_prob_refdist2_mean = []
        ref_dist_tensor: torch.Tensor = torch.zeros_like(model_output["dist"])
        for ref_dist in np.arange(self.mdn_threshold_eval - 0.5, self.mdn_threshold_eval + 0.01, 0.1, dtype=float):
            ref_dist_tensor.fill_(ref_dist)
            pair_prob_ref = calculate_probablity(model_output["pi"], model_output["sigma"], model_output["mu"], ref_dist_tensor)
            pair_prob_ref[cutoff_mask] = 0.
            pair_prob_ref_mean.append(pair_prob_ref)
            pair_prob_refdist2_mean.append(pair_prob_ref / (ref_dist ** 2))
        # previous reference computation method: probablity - reference probeblity
        if self.to_compute("MDN_SUM_REF"):
            sum_prob_ref = pair_prob_zeroed - pair_prob_ref_mean[-1]
            out_dict[f"MDN_SUM_REF"] = sum_prob_ref.view(-1)
        if self.to_compute("MDN_SUM_DIST2_REF"):
            sum_dist2_prob_ref = (pair_prob_zeroed - pair_prob_ref) / dist_sq
            out_dict[f"MDN_SUM_DIST2_REF"] = sum_dist2_prob_ref.view(-1)
        # new score method: score = ln(p/p_ref)
        # since we are computing log, the masked values should be 1.
        pair_prob_oned = torch.where(cutoff_mask, torch.ones_like(pair_prob), pair_prob)
        pair_prob_ref_mean = torch.stack(pair_prob_ref_mean, dim=-1).mean(dim=-1).view(-1) + 1e-9
        pair_prob_ref_mean[cutoff_mask] = 1.
        pair_prob_refdist2_mean = torch.stack(pair_prob_refdist2_mean, dim=-1).mean(dim=-1).view(-1) + 1e-9
        pair_prob_refdist2_mean[cutoff_mask] = 1.
        pair_log_ref_score = torch.log(pair_prob_oned) - torch.log(pair_prob_ref_mean)
        pair_log_refdist2_score = torch.log(pair_prob_oned) - torch.log(pair_prob_refdist2_mean)
        dist_sq[cutoff_mask] = 1.
        pair_log_dist2_ref_score = pair_log_ref_score - torch.log(dist_sq)
        pair_log_dist2_refdist2_score = pair_log_refdist2_score - torch.log(dist_sq)
        if self.to_compute("MDN_LOGSUM"): 
            out_dict["MDN_LOGSUM"] = torch.log(pair_prob_oned)
        if self.to_compute("MDN_LOGSUM_REF"):
            out_dict["MDN_LOGSUM_REF"] = pair_log_ref_score
        if self.to_compute("MDN_LOGSUM_DIST2"):
            out_dict["MDN_LOGSUM_DIST2"] = torch.log(pair_prob_oned) - torch.log(dist_sq)
        if self.to_compute("MDN_LOGSUM_DIST2_REF"):
            out_dict["MDN_LOGSUM_DIST2_REF"] = pair_log_dist2_ref_score
        if self.to_compute("MDN_LOGSUM_DIST2_REFDIST2"):
            out_dict["MDN_LOGSUM_DIST2_REFDIST2"] = pair_log_dist2_refdist2_score
        # TODO: reference alpha 1.57
        return out_dict
    
    def to_compute(self, key: str) -> bool:
        if self.requested_score_names is None: return True

        return key in self.requested_score_names
    
    def ext_mdn_scores_old(self, pair_prob: torch.Tensor, model_output: dict, data_batch, detail: dict) -> dict:
        """
        Update 08/10/2023: Deprecated since I found only reference works well. Other normalizations do not work.
        Computing external MDN scores such as normalized scores, max scores, scores subtracting references.
        """
        raise ValueError("Legacy code")
        out_dict = {}
        pair_prob = copy.deepcopy(pair_prob)
        ref_dist = copy.deepcopy(model_output["dist"])
        ref_dist.fill_(6.5)
        pair_prob_ref = calculate_probablity(model_output["pi"], model_output["sigma"], model_output["mu"], ref_dist)
        pair_batch = model_output["C_batch"].to(data_batch.N.device)
        for cutoff in [4.5, 3.5, 2.5, 1.5]:
            cutoff_mask = torch.where(model_output["dist"] > cutoff)[0]
            pair_prob[cutoff_mask] = 0.
            pair_prob_ref[cutoff_mask] = 0.
            # maximum probability
            max_prob = scatter_max(pair_prob, pair_batch, dim=0, dim_size=detail["n_units"])[0]
            out_dict[f"MDN_MAX_{cutoff}"] = max_prob.detach().cpu()

            # normalize by the number of atoms in the ligand
            sum_prob = scatter_add(pair_prob, pair_batch, dim=0, dim_size=detail["n_units"])
            out_dict[f"MDN_SUM_{cutoff}"] = sum_prob.view(-1)
            norm_prob = sum_prob.view(-1, 1) / data_batch.N.view(-1, 1)
            out_dict[f"MDN_NORM_{cutoff}"] = norm_prob.view(-1)
            sum_prob_ref = scatter_add(pair_prob - pair_prob_ref, pair_batch, dim=0, dim_size=detail["n_units"])
            out_dict[f"MDN_SUM__REF_{cutoff}"] = sum_prob_ref.view(-1)
            norm_prob_ref = sum_prob_ref.view(-1, 1) / data_batch.N.view(-1, 1)
            out_dict[f"MDN_NORM_REF_{cutoff}"] = norm_prob_ref.view(-1)
            # normalized by the number of heavy atoms in the ligand
            z_is_heavy = (data_batch.Z > 1).long()
            n_heavy = scatter_add(z_is_heavy, data_batch.atom_mol_batch, dim=0, dim_size=detail["n_units"])
            norm_prob = sum_prob.view(-1, 1) / n_heavy.view(-1, 1)
            out_dict[f"MDN_NORMHEAVY_{cutoff}"] = norm_prob.view(-1)
            norm_prob_ref = sum_prob_ref.view(-1, 1) / n_heavy.view(-1, 1)
            out_dict[f"MDN_NORMHEAVY_REF_{cutoff}"] = norm_prob_ref.view(-1)
        return out_dict

def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)

    return prob
