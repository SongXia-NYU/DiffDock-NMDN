from torch_cluster import radius_graph
from torch_scatter import scatter_min
import torch
import math
from torch_geometric.data import Data
from typing import List, Tuple

from dig.threedgraph.method import ComENet

class ComENetAtomEmbed(ComENet):
    def __init__(self, cfg: dict, out_channels=1):
        cutoff = cfg["comenet_cutoff"]
        num_layers = cfg["comenet_num_layers"]
        hidden_channels = middle_channels = cfg["n_feature"]
        num_radial = cfg["comenet_num_radial"]
        num_spherical = cfg["comenet_num_spherical"]
        num_output_layers = cfg["comenet_num_output_layers"]
        super().__init__(cutoff, num_layers, hidden_channels, middle_channels, out_channels, num_radial, num_spherical, num_output_layers)

        self.runtime_init: bool = False

    def forward(self, runtime_vars: dict):
        if not self.runtime_init:
            self.feature1.degreeInOrder = self.feature1.degreeInOrder.long()
            self.runtime_init = True
        data: Data = self.prepare_data(runtime_vars["data_batch"])
        atom_embed: torch.Tensor = self._forward(data)
        runtime_vars["comenet_atom_embed"] = atom_embed
        return runtime_vars

    @staticmethod
    def prepare_data(data_batch: Data) -> Data:
        # prepare our processed data for ComENet. Aka renaming variables.
        mappers: List[Tuple[str, str]] = [("Z", "z"), ("atom_mol_batch", "batch"), ("R", "pos")]
        for src_name, dst_name in mappers:
            setattr(data_batch, dst_name, getattr(data_batch, src_name))
        return data_batch
    
    def _forward(self, data):
        # The method is mainly copied from the parent method except for the readout layers and atom to mol properties.
        # ---------------- Copied from the parent method ---------------- #
        batch = data.batch
        z = data.z.long()
        pos = data.pos
        num_nodes = z.size(0)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        j, i = edge_index

        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        # Filter weird zero distances
        zero_dist_mask = (dist > 0.)
        edge_index = edge_index[:, zero_dist_mask]
        j, i = edge_index
        dist = dist[zero_dist_mask]
        vecs = pos[j] - pos[i]

        # Embedding block.
        x = self.emb(z)

        # Calculate distances.
        _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
        argmin0[argmin0 >= len(i)] = 0
        n0 = j[argmin0]
        add = torch.zeros_like(dist).to(dist.device)
        add[argmin0] = self.cutoff
        dist1 = dist + add

        _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
        argmin1[argmin1 >= len(i)] = 0
        n1 = j[argmin1]
        # --------------------------------------------------------

        _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
        argmin0_j[argmin0_j >= len(j)] = 0
        n0_j = i[argmin0_j]

        add_j = torch.zeros_like(dist).to(dist.device)
        add_j[argmin0_j] = self.cutoff
        dist1_j = dist + add_j

        # i[argmin] = range(0, num_nodes)
        _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
        argmin1_j[argmin1_j >= len(j)] = 0
        n1_j = i[argmin1_j]

        # ----------------------------------------------------------

        # n0, n1 for i
        n0 = n0[i]
        n1 = n1[i]

        # n0, n1 for j
        n0_j = n0_j[j]
        n1_j = n1_j[j]

        # tau: (iref, i, j, jref)
        # when compute tau, do not use n0, n0_j as ref for i and j,
        # because if n0 = j, or n0_j = i, the computed tau is zero
        # so if n0 = j, we choose iref = n1
        # if n0_j = i, we choose jref = n1_j
        mask_iref = n0 == j
        iref = torch.clone(n0)
        iref[mask_iref] = n1[mask_iref]
        idx_iref = argmin0[i]
        idx_iref[mask_iref] = argmin1[i][mask_iref]

        mask_jref = n0_j == i
        jref = torch.clone(n0_j)
        jref[mask_jref] = n1_j[mask_jref]
        idx_jref = argmin0_j[j]
        idx_jref[mask_jref] = argmin1_j[j][mask_jref]

        pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
            vecs,
            vecs[argmin0][i],
            vecs[argmin1][i],
            vecs[idx_iref],
            vecs[idx_jref]
        )

        # Calculate angles.
        a = ((-pos_ji) * pos_in0).sum(dim=-1)
        b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
        theta = torch.atan2(b, a)
        theta = torch.where(~ torch.isnan(theta), theta, torch.as_tensor(math.pi/2).to(theta.device).type(theta.type()))
        theta[theta < 0] = theta[theta < 0] + math.pi

        # Calculate torsions.
        dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
        plane1 = torch.cross(-pos_ji, pos_in0)
        plane2 = torch.cross(-pos_ji, pos_in1)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        phi = torch.atan2(b, a)
        phi = torch.where(~ torch.isnan(phi), phi, torch.as_tensor(math.pi/2).to(phi.device).type(phi.type()))
        phi[phi < 0] = phi[phi < 0] + math.pi

        # Calculate right torsions.
        plane1 = torch.cross(pos_ji, pos_jref_j)
        plane2 = torch.cross(pos_ji, pos_iref)
        a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
        b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
        tau = torch.atan2(b, a)
        tau = torch.where(~ torch.isnan(tau), tau, torch.as_tensor(math.pi/2).to(tau.device).type(tau.type()))
        tau[tau < 0] = tau[tau < 0] + math.pi

        feature1 = self.feature1(dist, theta, phi)
        feature2 = self.feature2(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature1, feature2, edge_index, batch)

        for lin in self.lins:
            x = self.act(lin(x))
        # ---------------- Copied from the parent method ---------------- #
        return x
