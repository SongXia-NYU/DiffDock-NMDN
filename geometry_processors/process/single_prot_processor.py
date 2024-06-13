import torch
import os.path as osp
from torch_geometric.data.data import Data
from geometry_processors.pl_dataset.ConfReader import PDBReader

from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.prot_utils import pp_min_dist_matrix_vec_mem


class SingleProtProcessor:
    def __init__(self, info: MPInfo, **reader_kwargs) -> None:
        self.info = info
        self.conf_reader_factory = ConfReaderFactory(info, **reader_kwargs)

    def process_single_entry(self) -> Data:
        sys_dict = {}
        pdb_reader: PDBReader = self.conf_reader_factory.get_prot_reader()[0]
        pad_dict: dict = pdb_reader.padding_style_dict
        sys_dict["res_num"] = torch.as_tensor(pad_dict["res_num"])
        sys_dict["R_prot_pad"] = torch.zeros((0, pad_dict["R"].shape[1], 3))
        sys_dict["Z_prot_pad"] = torch.zeros((0, pad_dict["R"].shape[1]))
        sys_dict["N_prot"] = torch.zeros((0, ))
        sys_dict["protein_file"] = pdb_reader.pdb_file
        if "pdb" not in self.info.labels:
            sys_dict["pdb"] = osp.basename(pdb_reader.pdb_file).split("_")[0]
        sys_dict.update(self.info.labels)

        # dummy variables for compatibility
        sys_dict["R"] = torch.zeros((0, 3))
        sys_dict["Z"] = torch.zeros((pad_dict["res_num"].shape[0], )).long()
        sys_dict["N"] = sys_dict["res_num"].shape[0]

        sys_dict["N_aa"] = pad_dict["N"].shape[0]
        # calculate protein intra-residue-residue min distance
        pp_min_dist = pp_min_dist_matrix_vec_mem(pad_dict["R"], pad_dict["R"])
        pp_min_dist = torch.as_tensor(pp_min_dist)
        # [num_edges, 2]
        edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_pp)
        # only record one-way edge
        edge_index_mask: torch.BoolTensor = (edge_index[:, 0] < edge_index[:, 1])
        edge_index = edge_index[edge_index_mask, :]
        pp_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]
        # [2, num_edges]
        edge_index = edge_index.T

        bond_type = f"PP_min_dist"
        sys_dict[f"{bond_type}_oneway_dist"] = pp_dist
        sys_dict[f"{bond_type}_oneway_edge_index"] = edge_index
        sys_dict["single_prot_identifier"] = 1
        return Data(**sys_dict)

