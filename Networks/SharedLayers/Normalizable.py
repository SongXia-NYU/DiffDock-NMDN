from typing import Union
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.utils_functions import floating_type

class Normalizable(ABC):
    def __init__(self, cfg: dict, energy_scale: Union[torch.Tensor, float]=None, energy_shift: Union[torch.Tensor, float]=None) -> None:
        # super().__init__()

        self.normalize: bool = cfg["normalize"]
        # atom level normalization
        if not self.normalize:
            return
        
        # remove the bias of self.lin_out
        self.lin_out = nn.Linear(cfg["n_feature"],cfg["n_output"], bias=False)
        self.lin_out.weight.data.zero_()

        # register normalization parameters
        ss_dim = cfg["n_output"]
        n_atom_embedding = cfg["n_atom_embedding"]
        shift_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type)
        scale_matrix = torch.zeros(n_atom_embedding, ss_dim).type(floating_type).fill_(1.0)
        if energy_shift is not None:
            if isinstance(energy_shift, torch.Tensor):
                shift_matrix[:, :] = energy_shift.view(1, -1)[:, :ss_dim]
            else:
                shift_matrix[:, 0] = energy_shift
        if energy_scale is not None:
            if isinstance(energy_scale, torch.Tensor):
                scale_matrix[:, :] = energy_scale.view(1, -1)[:, :ss_dim]
            else:
                scale_matrix[:, 0] = energy_scale
        self.register_parameter('scale', torch.nn.Parameter(scale_matrix, requires_grad=True))
        self.register_parameter('shift', torch.nn.Parameter(shift_matrix, requires_grad=cfg["train_shift"]))

    def norm_atom_prop(self, atom_prop: torch.Tensor, atomic_num: torch.LongTensor):
        if self.normalize:
            atom_prop = self.scale[atomic_num, :] * atom_prop + self.shift[atomic_num, :]
        return atom_prop
    
    @abstractmethod
    def register_parameter(self, name: str, param: torch.nn.Parameter):
        pass
