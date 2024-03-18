import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable

import numpy as np


class ConcreteDropout(nn.Module):
    """
    Adapted from https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-pytorch.ipynb
    Minor tweaks to be compatible with graph input
    """
    def __init__(self, module, weight_regularizer=1e-6, module_type='Linear',
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1, train_p=True, normal_dropout=False):
        """
        :param module:
        :param weight_regularizer:
        :param module_type: PhysNet | DimeNet | Linear
        :param dropout_regularizer:
        :param init_min:
        :param init_max:
        """
        super(ConcreteDropout, self).__init__()

        self.normal_dropout = normal_dropout
        self.module = module
        self.module_type = module_type
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        if self.normal_dropout:
            self.dropout = nn.Dropout(p=init_min)

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max), requires_grad=train_p)

    def freeze_prev_layers(self):
        self.module.freeze_prev_layers()

    def forward(self, *inputs):
        """

        :param inputs: has different property depending on PhysNet or DimeNet input.
        For PhysNet, input is: (node_representation, edge_index, expansion).
        For DimeNet, input: (msg_representation, rbf, sbf, msg_edge_index, edge_index).
        For Linear, input: (x)
        :return:
        """
        p = torch.sigmoid(self.p_logit)

        if self.normal_dropout:
            regularization = 0.
            module_out = self.module(*self._normal_dropout(inputs))
        else:
            module_out = self.module(*self._concrete_dropout(inputs, p))

            sum_of_square = 0
            for param in self.module.parameters():
                sum_of_square += torch.sum(torch.pow(param, 2))

            weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

            dropout_regularizer = p * torch.log(p)
            dropout_regularizer += (1. - p) * torch.log(1. - p)

            input_dimensionality = inputs[0][0].numel()  # Number of elements of first item in batch
            dropout_regularizer *= self.dropout_regularizer * input_dimensionality

            regularization = weights_regularizer + dropout_regularizer

        if self.module_type == 'Linear':
            return module_out, regularization
        else:
            return module_out[0], module_out[1], regularization

    def _concrete_dropout(self, inputs, p):
        """
        Dropout only apply to representation
        :param inputs:
        :param p:
        :return:
        """
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(inputs[0])

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(inputs[0], random_tensor)
        x = x / retain_prob

        if self.module_type == 'PhysNet':
            modified_input = (x, inputs[1], inputs[2])
        elif self.module_type == 'DimeNet':
            modified_input = (x, inputs[1], inputs[2], inputs[3], inputs[4])
        elif self.module_type == 'Linear':
            modified_input = (x, )
        else:
            raise ValueError('Unrecognised module type: {}'.format(self.module_type))

        return modified_input

    def _normal_dropout(self, inputs):
        if self.module_type == 'PhysNet':
            modified_input = (self.dropout(inputs[0]), inputs[1], inputs[2])
        elif self.module_type == 'DimeNet':
            modified_input = (self.dropout(inputs[0]), inputs[1], inputs[2], inputs[3], inputs[4])
        elif self.module_type == 'Linear':
            modified_input = (self.dropout(inputs[0]),)
        else:
            raise ValueError('Unrecognised module type: {}'.format(self.module_type))

        return modified_input
