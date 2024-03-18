import torch_geometric
from torch_geometric.nn.pool import MemPooling
import torch
import torch.nn as nn

from Networks.PhysLayers.PhysModule import OutputLayer


class MyMemPooling(nn.Module):
    def __init__(self, in_channels, n_output, extra_readout=False, **options):
        super(MyMemPooling, self).__init__()
        self.extra_readout = extra_readout
        self.n_output = n_output
        self.input_channel = in_channels

        if self.extra_readout:
            self.pool = MemPooling(in_channels, in_channels, **options)
            self.read_out = OutputLayer(in_channels, n_output, 1, "ssp", "none")
        else:
            self.pool = MemPooling(in_channels, n_output, **options)
            self.read_out = None

    def forward(self, **kwargs):
        out: torch.Tensor = self.pool(kwargs["vi"], kwargs["atom_mol_batch"])[0]
        out = torch.squeeze(out, dim=1)
        if self.read_out is not None:
            out = self.read_out(out)[0]
        return out
