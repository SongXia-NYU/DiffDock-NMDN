import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

class MartiniToAtomPooling(nn.Module):
    def __init__(self, reduce) -> None:
        super().__init__()
        self.reduce = reduce

    def forward(self, input_dict):
        h = input_dict["vi"]
        data_batch = input_dict["data_batch"]

        mol2lig_mask = (data_batch.mol_type == 1)
        mol2prot_mask = (data_batch.mol_type == 0)
        h_l = h[mol2lig_mask, :]
        h_p_martini = h[mol2prot_mask, :]
        h_p_aa = scatter(h_p_martini, data_batch.martini_aa_batch, dim=0, reduce=self.reduce, dim_size=data_batch.N_p_aa.sum())

        input_dict["h_l"] = h_l
        input_dict["h_p"] = h_p_aa
        return input_dict
