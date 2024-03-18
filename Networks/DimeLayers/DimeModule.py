import time

import torch
import torch.nn as nn

from Networks.DimeLayers.MessagePassingLayer import DimeNetMPN
from Networks.DimeLayers.OutputLayer import OutputLayer
from Networks.SharedLayers.ResidualLayer import ResidualLayer
from Networks.SharedLayers.ActivationFns import activation_getter
from utils.time_meta import record_data
from utils.utils_functions import floating_type, get_n_params


class DimeModule(nn.Module):
    def __init__(self, dim_rbf, dim_sbf, dim_msg, n_output, n_res_interaction, n_res_msg, n_dense_output, dim_bi_linear,
                 activation, uncertainty_modify):
        super().__init__()
        self.uncertainty_modify = uncertainty_modify
        self.activation = activation_getter(activation)

        msg_gate = torch.zeros(1, dim_msg).fill_(1.).type(floating_type)
        self.register_parameter('gate', nn.Parameter(msg_gate, requires_grad=True))

        self.message_pass_layer = DimeNetMPN(dim_bi_linear, dim_msg, dim_rbf, dim_sbf, activation)

        self.n_res_interaction = n_res_interaction
        for i in range(n_res_interaction):
            self.add_module('res_interaction{}'.format(i), ResidualLayer(dim_msg, activation))

        self.lin_interacted_msg = nn.Linear(dim_msg, dim_msg)

        self.n_res_msg = n_res_msg
        for i in range(n_res_msg):
            self.add_module('res_msg{}'.format(i), ResidualLayer(dim_msg, activation))

        self.output_layer = OutputLayer(dim_msg, dim_rbf, n_output, n_dense_output, activation,
                                        concrete_dropout=(uncertainty_modify == 'concreteDropoutOutput'))

    def forward(self, input_dict):
        msg_ji = input_dict["mji"]
        rbf_ji = input_dict["edge_attr"]["rbf_ji"]
        sbf_kji = input_dict["edge_attr"]["sbf_kji"]
        msg_edge_index = input_dict["msg_edge_index"]
        atom_edge_index = input_dict["edge_index"]

        # t0 = time.time()

        reserved_msg_ji = self.gate * msg_ji

        # t0 = record_data('main.gate', t0)

        mji = self.message_pass_layer(msg_ji, msg_edge_index, rbf_ji, sbf_kji)

        # t0 = record_data('main.message-passing', t0)

        for i in range(self.n_res_interaction):
            mji = self._modules['res_interaction{}'.format(i)](mji)

        # t0 = record_data('main.res-layer', t0)

        mji = self.activation(self.lin_interacted_msg(mji))

        mji = mji + reserved_msg_ji

        # t0 = record_data('main.lin-layer', t0)

        for i in range(self.n_res_msg):
            mji = self._modules['res_msg{}'.format(i)](mji)

        # t0 = record_data('main.res-layer-2', t0)

        out, regularization = self.output_layer(mji, rbf_ji, atom_edge_index)

        # t0 = record_data('main.out-layer', t0)
        return {"mji": mji, "out": out, "regularization": regularization}

    def freeze_prev_layers(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.output_layer.parameters():
            param.requires_grad_(True)
        return


if __name__ == '__main__':
    model = DimeModule(6, 36, 128, 12, 1, 2, 3, 8, 'swish')
    print(get_n_params(model) - get_n_params(model.output_layer))
