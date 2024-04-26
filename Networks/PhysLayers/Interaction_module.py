import torch
import torch.nn as nn
import torch.nn.functional
import torch_geometric
from Networks.PairedPropLayers.MDNLayer import get_prot_dim

from utils.utils_functions import semi_orthogonal_glorot_weights, floating_type
from Networks.SharedLayers.ResidualLayer import ResidualLayer
from Networks.SharedLayers.ActivationFns import activation_getter


class InteractionModule(nn.Module):
    """
    The interaction layer defined in PhysNet
    """

    def __init__(self, F, K, n_res_interaction, activation, preserve_prot_embed, batch_norm, dropout, config_dict, embed_name, module_str):
        super().__init__()
        u = torch.Tensor(1, F).type(floating_type).fill_(1.)
        self.register_parameter('u', torch.nn.Parameter(u, True))

        mpnn_cls = MessagePassingLayer
        if module_str in ["P-PL", "P-PL-noOut"]:
            mpnn_cls = PLMessagePassingLayer
        if module_str in ["P-PM-noOut"]:
            mpnn_cls = PMMessagePassingLayer

        self.message_pass_layer = mpnn_cls(aggr='add', F=F, K=K, activation=activation, embed_name=embed_name,
                                           batch_norm=batch_norm, config_dict=config_dict)

        self.n_res_interaction = n_res_interaction
        for i in range(n_res_interaction):
            self.add_module('res_layer' + str(i), ResidualLayer(F=F, activation=activation, batch_norm=batch_norm,
                                                                dropout=dropout))

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(F, momentum=1.)
        self.lin_last = nn.Linear(F, F)
        if preserve_prot_embed:
            self.lin_last.weight.data.zero_()
        else:
            self.lin_last.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_last.bias.data.zero_()

        self.activation = activation_getter(activation)
        self.embed_name = embed_name

    def forward(self, input_dict):
        x = input_dict[self.embed_name]
        msged_x = self.message_pass_layer(input_dict)
        tmp_res = msged_x
        for i in range(self.n_res_interaction):
            tmp_res = self._modules['res_layer' + str(i)](tmp_res)
        if self.batch_norm:
            tmp_res = self.bn(tmp_res)
        v = self.activation(tmp_res)
        v = self.lin_last(v)
        return v + torch.mul(x, self.u), msged_x


class MessagePassingLayer(torch_geometric.nn.MessagePassing):
    """
    message passing layer in torch_geometric
    see: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html for more details
    """

    def __init__(self, F, K, activation, aggr, batch_norm, config_dict, embed_name, flow = 'source_to_target'):
        self.batch_norm = batch_norm
        super().__init__(aggr=aggr, flow=flow)
        self.config_dict = config_dict
        self.lin_for_same = nn.Linear(F, F)
        self.lin_for_same.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_same.bias.data.zero_()

        self.lin_for_diff = nn.Linear(F, F)
        self.lin_for_diff.weight.data = semi_orthogonal_glorot_weights(F, F)
        self.lin_for_diff.bias.data.zero_()

        if self.batch_norm:
            self.bn_same = nn.BatchNorm1d(F, momentum=1.)
            self.bn_diff = nn.BatchNorm1d(F, momentum=1.)

        self.G = nn.Linear(K, F, bias=False)
        self.G.weight.data.zero_()

        self.activation = activation_getter(activation)
        self.embed_name = embed_name

    def message(self, x_j, edge_attr):
        if self.batch_norm:
            x_j = self.bn_diff(x_j)
        msg = self.lin_for_diff(x_j)
        msg = self.activation(msg)
        masked_edge_attr = self.G(edge_attr)
        msg = torch.mul(msg, masked_edge_attr)
        return msg

    def forward(self, input_dict):
        x = input_dict[self.embed_name]
        edge_index = input_dict["edge_index"]
        edge_attr = input_dict["edge_attr"]["rbf"]

        x = self.activation(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def update(self, aggr_out, x):
        if self.batch_norm:
            x = self.bn_same(x)
        a = self.activation(self.lin_for_same(x))
        return a + aggr_out


class HeteroMessagePassingLayer(MessagePassingLayer):
    """
    Message passing on a heterogeous graph with two different nodes
    """
    def __init__(self, f_node1: int, f_node2: int, f_rbf: int, activation: str, aggr: str, batch_norm: bool,
                cfg: dict, name1: str, name2: str, flow="target_to_source"):
        super().__init__(f_node1, f_rbf, activation, aggr, batch_norm, cfg, None, flow)

        self.lin_for_diff = nn.Linear(f_node2, f_node1)
        self.lin_for_diff.weight.data = semi_orthogonal_glorot_weights(f_node2, f_node1)
        self.lin_for_diff.bias.data.zero_()
        
        self.name1: str = name1
        self.name2: str = name2

    def forward(self, runtime_vars: dict):
        # [n_node1, f_node1]
        x_1 = runtime_vars[self.name1]
        x_1 = self.activation(x_1)
        # [n_node2, f_node2]
        x_2 = runtime_vars[self.name2]
        x_2 = self.activation(x_2)
        edge_index = runtime_vars["edge_index"]
        edge_attr = runtime_vars["edge_attr"]["rbf"]
        return self.propagate(edge_index, x=(x_1, x_2), size=(x_1.shape[0], x_2.shape[0]), edge_attr=edge_attr)

    def update(self, aggr_out, x):
        x_lig, x_prot = x
        if self.batch_norm:
            x = self.bn_same(x)
        a = self.activation(self.lin_for_same(x_lig))
        return a + aggr_out
    
    def message(self, x_j, edge_attr):
        msg = super().message(x_j, edge_attr)
        return msg


class PLMessagePassingLayer(HeteroMessagePassingLayer):
    """
    MessgePassing from protein node to ligand node. 
    The PL graph is treated as a bipartite graph because they have different dimensions.
    """
    def __init__(self, F, K, activation, aggr, batch_norm, config_dict, embed_name):
        assert embed_name == "vi", embed_name
        prot_dim = get_prot_dim(config_dict)
        # flow is "target_to_source" because we want information to flow from protein to ligand
        super().__init__(F, prot_dim, K, activation, aggr, batch_norm, config_dict, 
                         embed_name, "prot_embed", flow="target_to_source")


class PMMessagePassingLayer(HeteroMessagePassingLayer):
    """
    MessgePassing from protein node to metal node. 
    The PL graph is treated as a bipartite graph because they have different dimensions.
    """
    def __init__(self, F, K, activation, aggr, batch_norm, config_dict, embed_name):
        assert embed_name == "kano_atom_embed", embed_name
        assert F == 300, F
        prot_dim = get_prot_dim(config_dict)
        # encoded by KANO
        metal_dim = 300
        # flow is "target_to_source" because we want information to flow from protein to ligand
        super().__init__(metal_dim, prot_dim, K, activation, aggr, batch_norm, config_dict, 
                         "kano_atom_embed", "prot_embed", flow="target_to_source")


if __name__ == '__main__':
    pass
