from typing import Union
import torch
import torch.nn as nn
from torch_scatter import scatter
from Networks.SharedLayers.Normalizable import Normalizable

from Networks.comenet.ComENetAtomEmbed import ComENetAtomEmbed
from utils.utils_functions import floating_type


class ComENetMolProp(ComENetAtomEmbed, Normalizable):
    """
    Predicting molecular level property.
    """
    def __init__(self, cfg: dict, energy_scale: Union[torch.Tensor, float]=None, energy_shift: Union[torch.Tensor, float]=None):
        ComENetAtomEmbed.__init__(self, cfg, cfg["n_output"])
        Normalizable.__init__(self, cfg, energy_scale, energy_shift)

    def forward(self, runtime_vars: dict):
        batch = runtime_vars["data_batch"].atom_mol_batch
        runtime_vars = super().forward(runtime_vars)
        atom_embed = runtime_vars["comenet_atom_embed"]

        atom_prop = self.lin_out(atom_embed)
        # atom level normalization
        atom_prop = self.norm_atom_prop(atom_prop, runtime_vars["data_batch"].Z)
        mol_prop = scatter(atom_prop, batch, dim=0)
        runtime_vars["comenet_mol_prop"] = mol_prop
        return runtime_vars
