import torch
import torch.nn as nn

class CombinedPKano(nn.Module):
    def __init__(self) -> None:
        # combining sPhysNet atom embedding with KANO atom embedding.
        super().__init__()

    def forward(self, runtime_vars: dict):
        kano_atom_embed: torch.Tensor = runtime_vars["kano_atom_embed"]
        sphysnet_embed: torch.Tensor = runtime_vars["vi"]

        combined_embed: torch.Tensor = torch.concat([sphysnet_embed, kano_atom_embed], dim=-1)
        runtime_vars["kano_atom_embed"] = combined_embed
        return runtime_vars
