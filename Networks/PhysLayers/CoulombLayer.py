import torch
import torch.nn as nn

from utils.utils_functions import cal_coulomb_E, floating_type


class CoulombLayer(nn.Module):
    """
    This layer is used to calculate atom-wise coulomb interaction
    """
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = torch.as_tensor(cutoff).type(floating_type)

    def forward(self, qi, edge_dist, edge_index, q_ref=None, N=None, atom_mol_batch=None):
        return cal_coulomb_E(qi, edge_dist, edge_index, self.cutoff, q_ref=q_ref, N=N, atom_mol_batch=atom_mol_batch)
