import torch
import torch.nn as nn
from torch_scatter import scatter
from Networks.PhysLayers.PhysModule import OutputLayer

from Networks.SharedLayers.ActivationFns import activation_getter
"""
First do pooling from atom embedding to molecule embedding, then readout
"""
class PoolingReadout(nn.Module):
    def __init__(self, activation: str, pooling_type: str = "mean", readout_type: str = "lin", **readout_kwargs) -> None:
        super().__init__()

        readout_factory = ReadoutLayerFactory()
        self.readout_layer = readout_factory.get_layer(readout_type, activation=activation, **readout_kwargs)
        self.pooling_type: str = pooling_type

    def forward(self, runtime_vars: dict):
        atom_embed: torch.Tensor = runtime_vars["vi"]
        atom_mol_batch: torch.LongTensor = runtime_vars["data_batch"].atom_mol_batch
        if self.pooling_type in ["mean", "sum"]:
            mol_embed: torch.Tensor = scatter(reduce=self.pooling_type, src=atom_embed, index=atom_mol_batch, dim=0)
        else:
            raise NotImplementedError(self.pooling_type)
        mol_prop = self.readout_layer(mol_embed)
        runtime_vars["pooling_reaout_mol_prop"] = mol_prop
        return runtime_vars

class LinReadout(nn.Module):
    def __init__(self, activation: str, n_feature: int, n_output: int, dropout: float = 0.0, batch_norm: bool = True) -> None:
        super().__init__()

        self.layers = nn.Sequential(
                nn.Linear(n_feature, n_feature),
                nn.BatchNorm1d(n_feature) if batch_norm else nn.Identity(),
                activation_getter(activation),
                nn.Dropout(dropout),
                nn.Linear(n_feature, n_feature),
                nn.BatchNorm1d(n_feature) if batch_norm else nn.Identity(),
                activation_getter(activation),
                nn.Dropout(dropout),
                nn.Linear(n_feature, n_output)
            )
    def forward(self, data: torch.Tensor):
        return self.layers(data)
    

class PhysOutputLayer(OutputLayer):
    def forward(self, x):
        return super().forward(x)[0]


class ReadoutLayerFactory:
    def __init__(self) -> None:
        pass

    def get_layer(self, readout_type: str, **m_args) -> nn.Module:
        if readout_type == "phys_out":
            return PhysOutputLayer(m_args["n_feature"], m_args["n_output"], m_args["n_res_output"], 
                m_args["activation"], m_args["uncertainty_modify"])
        
        if readout_type == "identity":
            return nn.Identity()
    
        for key in ["n_res_output", "uncertainty_modify"]:
            del m_args[key]
        assert readout_type == "lin", readout_type
        return LinReadout(**m_args)
