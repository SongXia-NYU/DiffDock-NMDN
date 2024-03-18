import torch
import torch.nn as nn

from Networks.SharedLayers.ActivationFns import activation_getter
from Networks.PairedPropLayers.MDNLayer import get_prot_dim, overwrite_prot_dim
"""
Transform protein embedding into lower dimension
"""

class ProtEmbedTransformLayer(nn.Module):
    def __init__(self, config_dict: dict, out_dim: int, activation: str, 
                 tranform="linear", in_dim=None, bias=False) -> None:
        super().__init__()
        if in_dim is None:
            in_dim = get_prot_dim(config_dict, want_original=True)
        layers = []
        if tranform == "linear":
            layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            layers.append(activation_getter(activation))
        else:
            raise NotImplementedError
        self.layers = nn.Sequential(*layers)
        # overwrite the protein dimesion so other layers have the correct dimension.
        overwrite_prot_dim(config_dict, out_dim)

    def forward(self, input_dict: dict):
        h_p = input_dict["prot_embed"]
        transformed = self.layers(h_p)
        input_dict["prot_embed"] = transformed
        return input_dict
