import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_functions import get_device


def _shifted_soft_plus(x):
    """
    activation function in PhysNet: shifted softplus function
    sigma(x) = log (exp(x) + 1) − log (2)
    :param x:
    :return:
    """
    return nn.functional.softplus(x) - torch.Tensor([np.log(2)]).type(x.type()).to(x.device)


class ShiftedSoftPlus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _shifted_soft_plus(x)


class Swish(nn.Module):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


# """
# Change activation function here
# May be written into config file later
# """
# current_activation_fn = Swish()


activation_cls_mapper = {
    'swish': Swish,
    'shifted_soft_plus': ShiftedSoftPlus,
    'shiftedsoftplus': ShiftedSoftPlus,
    'ssp': ShiftedSoftPlus,
    "relu": nn.ReLU,
    "identity": nn.Identity
}


def activation_getter(string_activation):
    if string_activation.lower() in activation_cls_mapper.keys():
        return activation_cls_mapper[string_activation.lower()]()
    else:
        print('No activation function named {},'
              ' only those are available: {}. exiting...'.format(string_activation, activation_cls_mapper.keys()))
        exit()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    _x = torch.arange(-5, 5, 0.01)
    swish = Swish()
    y = swish(_x)
    plt.plot(_x, y)
    plt.vlines(0, y.min(), y.max())
    plt.hlines(0, _x.min(), _x.max())
    plt.show()
