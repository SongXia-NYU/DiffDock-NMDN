from torch_cluster import radius_graph
import torch
from torch_geometric.data import Data
from torch_scatter import scatter


from dig.threedgraph.method import SphereNet
from dig.threedgraph.utils import xyz_to_dat

from Networks.comenet.ComENetAtomEmbed import ComENetAtomEmbed


class SphereNetWrapped(SphereNet):
    def __init__(self, cfg: dict):
        init_kwargs: dict = {}
        init_kwargs["out_channels"] = cfg["n_output"]
        # hyperparams obtained from https://github.com/divelab/DIG/blob/dig-stable/examples/threedgraph/threedgraph.ipynb
        init_kwargs["energy_and_force"] = False
        init_kwargs["cutoff"] = 5.0
        init_kwargs["num_layers"] = 4
        init_kwargs["hidden_channels"] = 128
        init_kwargs["int_emb_size"] = 64
        init_kwargs["basis_emb_size_dist"] = 8
        init_kwargs["basis_emb_size_angle"]=8
        init_kwargs["basis_emb_size_torsion"]=8
        init_kwargs["out_emb_channels"]=256
        init_kwargs["num_spherical"]=3
        init_kwargs["num_radial"]=6
        init_kwargs["envelope_exponent"]=5
        init_kwargs["num_before_skip"]=1
        init_kwargs["num_after_skip"]=2
        init_kwargs["num_output_layers"]=3
        init_kwargs["use_node_features"]=True
        super().__init__(**init_kwargs)

    def forward(self, runtime_vars: dict):
        data: Data = ComENetAtomEmbed.prepare_data(runtime_vars["data_batch"])
        edge_embed, atom_embed, mol_prop = self._forward(data)
        runtime_vars["spherenet_atom_embed"] = atom_embed
        runtime_vars["spherenet_mol_prop"] = mol_prop
        return runtime_vars
    
    def _forward(self, batch_data):
        # -----------------copied from parent forward method----------------- #
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        num_nodes=z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)
        # -----------------End of copy----------------- #
        __, e2 = e
        atom_embed: torch.Tensor = scatter(e2, i, dim=0)
        # edge embedding; atom embedding and mol properties
        return e, atom_embed, u
