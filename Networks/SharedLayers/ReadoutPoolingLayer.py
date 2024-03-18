import torch
import torch.nn as nn
from torch_scatter import scatter

from Networks.SharedLayers.PoolingReadoutLayer import LinReadout, ReadoutLayerFactory
"""
First do atom-level readout layer and then do pooling
"""
class ReadoutPooling(nn.Module):
    def __init__(self, activation: str, pooling_type: str = "mean", readout_type: str = "lin", **readout_kwargs) -> None:
        super().__init__()

        readout_factory = ReadoutLayerFactory()
        self.readout_layer = readout_factory.get_layer(readout_type, activation=activation, **readout_kwargs)
        self.pooling_type: str = pooling_type

    def forward(self, runtime_vars: dict):
        atom_embed: torch.Tensor = runtime_vars["vi"]
        atom_prop = self.readout_layer(atom_embed)

        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        if self.pooling_type in ["mean", "sum"]:
            mol_prop: torch.Tensor = scatter(reduce=self.pooling_type, src=atom_prop, index=atom_mol_batch, dim=0)
        else:
            raise NotImplementedError(self.pooling_type)
        runtime_vars["readout_pooling_mol_prop"] = mol_prop
        return runtime_vars
