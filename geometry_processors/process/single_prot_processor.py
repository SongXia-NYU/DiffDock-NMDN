import torch
import os.path as osp
from torch_geometric.data import HeteroData
from geometry_processors.DataPrepareUtils import pairwise_dist
from geometry_processors.pl_dataset.ConfReader import PDBReader

from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.prot_utils import pp_min_dist_matrix_vec_mem


class SingleProtProcessor:
    def __init__(self, info: MPInfo, **reader_kwargs) -> None:
        self.info = info
        assert abs(info.cutoff_pp-15.) < 1e-3
        self.conf_reader_factory = ConfReaderFactory(info, **reader_kwargs)

    def process_single_entry(self) -> HeteroData:
        sys_dict = {}
        sys_dict["single_prot_identifier"] = 1

        prot_dict = {}

        pdb_reader: PDBReader = self.conf_reader_factory.get_prot_reader()[0]
        pad_dict: dict = pdb_reader.padding_style_dict
        prot_dict["res_num"] = torch.as_tensor(pad_dict["res_num"])
        prot_dict["R_prot_pad"] = torch.zeros((0, pad_dict["R"].shape[1], 3))
        prot_dict["Z_prot_pad"] = torch.zeros((0, pad_dict["R"].shape[1]))
        prot_dict["N_prot"] = torch.zeros((0, ))
        sys_dict["protein_file"] = pdb_reader.pdb_file
        if "pdb" not in self.info.labels:
            sys_dict["pdb"] = osp.basename(pdb_reader.pdb_file).split("_")[0]
        sys_dict.update(self.info.labels)

        # dummy variables for compatibility
        prot_dict["R"] = torch.zeros((0, 3))
        prot_dict["Z"] = torch.zeros((pad_dict["res_num"].shape[0], )).long()
        prot_dict["N"] = prot_dict["res_num"].shape[0]
        prot_dict["num_nodes"] = prot_dict["res_num"].shape[0]

        prot_dict["N_aa"] = pad_dict["N"].shape[0]
        sys_dict = {"protein": prot_dict}
        res = HeteroData(**sys_dict)

        # calculate protein intra-residue-residue min distance
        pp_min_dist, pp_max_dist = pp_min_dist_matrix_vec_mem(pad_dict["R"], pad_dict["R"], True)
        pp_min_dist = torch.as_tensor(pp_min_dist)
        # [num_edges, 2]
        edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_pp)
        # only record one-way edge
        edge_index_mask: torch.BoolTensor = (edge_index[:, 0] < edge_index[:, 1])
        edge_index = edge_index[edge_index_mask, :]
        pp_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]
        # [2, num_edges]
        edge_index = edge_index.T
        bond_type = ("protein", "min_dist", "protein")
        res[bond_type].dist = pp_dist
        res[bond_type].edge_index = edge_index

        # calculate protein intra-residue-residue max distance
        pp_max_dist = torch.as_tensor(pp_max_dist)
        # [num_edges, 2]
        edge_index = torch.nonzero(pp_max_dist<=self.info.cutoff_pp)
        # only record one-way edge
        edge_index_mask: torch.BoolTensor = (edge_index[:, 0] < edge_index[:, 1])
        edge_index = edge_index[edge_index_mask, :]
        pp_dist = pp_max_dist[edge_index[:, 0], edge_index[:, 1]]
        # [2, num_edges]
        edge_index = edge_index.T
        bond_type = ("protein", "max_dist", "protein")
        res[bond_type].dist = pp_dist
        res[bond_type].edge_index = edge_index

        # c-alpha distance
        c_alpha_coords = pdb_reader.c_alpha_coords()
        c_alpha_coords = torch.as_tensor(c_alpha_coords)
        pair_dist = pairwise_dist(c_alpha_coords, c_alpha_coords)
        edge_index = torch.nonzero(pair_dist<=self.info.cutoff_pp)
        edge_index = edge_index.T
        # convert into one_way interaction
        edge_index = edge_index[:, (edge_index[0] < edge_index[1])]
        pair_dist = pair_dist[edge_index[0], edge_index[1]]
        bond_type = ("protein", "c_alpha_dist", "protein")
        res[bond_type].dist = pair_dist
        res[bond_type].edge_index = edge_index

        c_beta_coords = pdb_reader.c_beta_coords()
        c_beta_coords = torch.as_tensor(c_beta_coords)
        pair_dist = pairwise_dist(c_beta_coords, c_beta_coords)
        edge_index = torch.nonzero(pair_dist<=self.info.cutoff_pp)
        edge_index = edge_index.T
        # convert into one_way interaction
        edge_index = edge_index[:, (edge_index[0] < edge_index[1])]
        pair_dist = pair_dist[edge_index[0], edge_index[1]]
        bond_type = ("protein", "c_beta_dist", "protein")
        res[bond_type].dist = pair_dist
        res[bond_type].edge_index = edge_index

        return res

