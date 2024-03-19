import torch
from torch_geometric.data.data import Data

from geometry_processors.lazy_property import lazy_property
from geometry_processors.pl_dataset.ConfReader import ConfReader, PDBReader
from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.prot_utils import pp_min_dist_matrix_vec_mem

class ProtProtProcessor:
    def __init__(self, info: MPInfo, **reader_kwargs) -> None:
        self.info = info
        self.conf_reader_factory = ConfReaderFactory(info, **reader_kwargs)

    @lazy_property
    def chain1_reader(self) -> PDBReader:
        return self.conf_reader_factory.get_chain1_reader()[0]
    
    @lazy_property
    def chain2_reader(self) -> PDBReader:
        return self.conf_reader_factory.get_chain2_reader()[0]
    
    def process_single_entry(self) -> Data:
        sys_dict = {}
        chain1_pad_dict: dict = self.chain1_reader.padding_style_dict
        chain2_pad_dict: dict = self.chain2_reader.padding_style_dict

        sys_dict["N_aa_chain1"] = chain1_pad_dict["N"].shape[0]
        sys_dict["N_aa_chain2"] = chain2_pad_dict["N"].shape[0]

        # calculate protein residue-residue min distance
        pp_min_dist = pp_min_dist_matrix_vec_mem(chain1_pad_dict["R"], chain2_pad_dict["R"])
        pp_min_dist = torch.as_tensor(pp_min_dist)
        edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_pp)
        pp_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]

        bond_type = "PP_min_dist"
        # the one-way edges are used by MDN layers
        sys_dict[f"{bond_type}_oneway_dist"] = pp_dist
        sys_dict[f"{bond_type}_oneway_edge_index"] = edge_index.T
        sys_dict = self.infuse_labels(sys_dict)

        # add external file information
        for ext_info_name in ["dockground2_combined_pdb", "protein_chain1_pdb", "protein_chain2_pdb",
                              "chains_combined_pdb"]:
            if getattr(self.info, ext_info_name) is None: continue
            sys_dict[ext_info_name] = getattr(self.info, ext_info_name)

        return Data(**sys_dict)

    def infuse_labels(self, system_dict: dict) -> dict:
        for key in self.info.labels:
            val = self.info.labels[key]
            if isinstance(val, (int, float, bool)):
                system_dict[key] = torch.as_tensor(val).view(-1)
            else:
                assert isinstance(val, str), f"unsopported type:{val}, {val.__class__}"
                system_dict[key] = val
        return system_dict


class ProtProtIntraProcessor(ProtProtProcessor):
    # Includes both inter- and intra- molecular interaction
    def __init__(self, info: MPInfo, **reader_kwargs) -> None:
        super().__init__(info, **reader_kwargs)

    def process_single_entry(self) -> Data:
        data_inter: Data = super().process_single_entry()
        for chain_num in [1, 2]:
            chain_pad_dict = getattr(self, f"chain{chain_num}_reader").padding_style_dict
            # calculate protein intra-residue-residue min distance
            pp_min_dist = pp_min_dist_matrix_vec_mem(chain_pad_dict["R"], chain_pad_dict["R"])
            pp_min_dist = torch.as_tensor(pp_min_dist)
            # [num_edges, 2]
            edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_pp)
            pp_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]
            # [2, num_edges]
            edge_index = edge_index.T
            # [numres, ]
            resnum: torch.LongTensor = torch.as_tensor(chain_pad_dict["res_num"]).long().view(-1)
            # [num_edges, ]
            resnum_i: torch.LongTensor = resnum[edge_index[0]]
            resnum_j: torch.LongTensor = resnum[edge_index[1]]
            # masking self interaction and neighbor edges
            non_neighbor: torch.BoolTensor = ((resnum_i - resnum_j).abs() >= 2)
            # record edges and distances
            edge_index = edge_index[:, non_neighbor]
            pp_dist = pp_dist[non_neighbor]

            bond_type = f"PP_min_dist_chain{chain_num}"
            setattr(data_inter, f"{bond_type}_oneway_dist", pp_dist)
            setattr(data_inter, f"{bond_type}_oneway_edge_index", edge_index)
        return data_inter

