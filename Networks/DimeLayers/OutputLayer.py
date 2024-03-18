import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from Networks.SharedLayers.ActivationFns import activation_getter
from Networks.UncertaintyLayers.MCDropout import ConcreteDropout
from utils.utils_functions import get_n_params


class _MPNScatter(MessagePassing):
    """
    Message passing layer exclusively used for scatter_
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        return aggr_out + x


class OutputLayer(torch.nn.Module):
    """
    The output layer(red one in paper) of DimeNet
    """
    def __init__(self, embedding_dim, rbf_dim, n_output, n_dense, activation, concrete_dropout):
        super().__init__()
        self.concrete_dropout = concrete_dropout
        self.embedding_dim = embedding_dim
        self.rbf_dim = rbf_dim

        self.activation = activation_getter(activation)
        self.n_dense = n_dense
        for i in range(n_dense):
            if self.concrete_dropout:
                self.add_module('dense{}'.format(i),
                                ConcreteDropout(nn.Linear(embedding_dim, embedding_dim), module_type='Linear'))
            else:
                self.add_module('dense{}'.format(i), nn.Linear(embedding_dim, embedding_dim))
        self.lin_rbf = nn.Linear(rbf_dim, embedding_dim, bias=False)

        self.scatter_fn = _MPNScatter()

        self.out_dense = nn.Linear(embedding_dim, n_output, bias=False)
        self.out_dense.weight.data.zero_()
        if self.concrete_dropout:
            self.out_dense = ConcreteDropout(self.out_dense, module_type='Linear')

    def forward(self, m_ji, rbf_ji, atom_edge_index):
        regularization = 0.
        # t0 = time.time()

        e_ji = self.lin_rbf(rbf_ji)

        # t0 = record_data('main.output.lin-rbf', t0)

        message_ji = e_ji * m_ji

        # t0 = record_data('main.output.cal-msg', t0)

        '''
        message to atomic information
        '''
        atom_i = scatter(reduce='add', src=message_ji, index=atom_edge_index[1, :], dim=-2)

        # t0 = record_data('main.output.merge-msg', t0)

        for i in range(self.n_dense):
            if self.concrete_dropout:
                atom_i, reg = self._modules['dense{}'.format(i)](atom_i)
                regularization = regularization + reg
            else:
                atom_i = self._modules['dense{}'.format(i)](atom_i)
            atom_i = self.activation(atom_i)

        # t0 = record_data('main.output.dense', t0)

        if self.concrete_dropout:
            out, reg = self.out_dense(atom_i)
            regularization = regularization + reg
        else:
            out = self.out_dense(atom_i)

        # t0 = record_data('main.output.out', t0)
        return out, regularization


if __name__ == '__main__':
    model = OutputLayer(128, 6, 12, 3, 'swish')
    print(get_n_params(model))
