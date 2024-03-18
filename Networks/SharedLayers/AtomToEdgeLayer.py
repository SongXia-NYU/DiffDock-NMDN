import torch
import torch.nn as nn

from Networks.SharedLayers.ActivationFns import activation_getter


class AtomToEdgeLayer(nn.Module):
    def __init__(self, dim_rbf, dim_edge, activation):
        super().__init__()
        self.lin_rbf = nn.Linear(dim_rbf, dim_edge)
        self.lin_concat = nn.Linear(dim_edge * 3, dim_edge)

        self.activation = activation_getter(activation)

    def forward(self, input_dict):
        atom_attr = input_dict["vi"]
        edge_index = input_dict["edge_index"]
        rbf = input_dict["edge_attr"]["rbf"]
        h_i = atom_attr[edge_index[0, :], :]
        h_j = atom_attr[edge_index[1, :], :]

        concat_msg = torch.cat([self.lin_rbf(rbf), h_j, h_i], dim=-1)
        m_ji = self.activation(self.lin_concat(concat_msg))

        return {"mji": m_ji}
