from utils.DataPrepareUtils import pairwise_dist
from utils.configs import Config
from utils.data.MyData import MyData
from utils.data.data_utils import get_lig_coords
from utils.utils_functions import floating_type, get_device, lazy_property


import torch
from torch_geometric.data import HeteroData


from typing import Dict, Set, Union


class DataPostProcessor:
    """
    Process protein-ligand data on the fly. Including:
    - cut proteins into pockets
    - calculate edge on the fly
    """
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg is None: return
        # it saves all bond types that are needed to be computed on the fly.
        self.bond_type_need_compute = set()
        self.bond_type_need_concat = set()

        self._drop = set()
        self.proc_in_gpu = cfg.data.proc_in_gpu

        self._edge_precalculated = None
        self._compute_n_pl = None

        self._cutoff_query = None
        self.cache_bonds: bool = cfg.data.cache_bonds
        if self.cache_bonds: self.bond_cache_query: Dict[str, Dict[str, torch.Tensor]] = {}

    @lazy_property
    def bonding_types(self) -> Set[str]:
        # a set containing all of the bond types needed for the run.
        # the processor will check if they are already provided or not. 
        # if not, they will be calculated on the fly.
        btype_list = self.cfg.model.bonding_type.split()
        bonding_types = set([s for s in btype_list if s.lower()!="none"]) if self.cfg is not None else None
        # for backward compatibility
        if self.cfg.training.loss_fn.loss_metric == "mdn" and btype_list[-1].lower() == "none":
            bonding_types.add("PL")
        return bonding_types
    
    def proc_hetero(self, data: HeteroData):
        data["ligand"].num_nodes = data["ligand"].N
        # protein: number of protein residues
        data["protein"].num_nodes = data["protein"].N.shape[0]
        data["ion"].num_nodes = data["ion"].Z.shape[0]
        data["water"].num_nodes = data["water"].Z.shape[0]

        data[("ligand", "interaction", "protein")].edge_index = data[("ligand", "interaction", "protein")].min_dist_edge_index
        data[("ligand", "interaction", "ion")].edge_index = data[("ligand", "interaction", "ion")].min_dist_edge_index
        data[("ligand", "interaction", "water")].edge_index = data[("ligand", "interaction", "water")].min_dist_edge_index
        
        lig_coords = get_lig_coords(data)
        pair_dist = pairwise_dist(lig_coords, lig_coords)
        edge_index = torch.nonzero(pair_dist<self.cutoff_query["BN"])
        edge_index = edge_index.T
        # remove self interaction
        edge_index = edge_index[:, (edge_index[0] != edge_index[1])]
        data[("ligand", "interaction", "ligand")].edge_index = edge_index
        return data

    def __call__(self, data: Union[MyData, HeteroData], idx: int):
        if self.cfg is None:
            return data

        if self.proc_in_gpu:
            data = data.to(get_device())

        if isinstance(data, HeteroData):
            return self.proc_hetero(data)

        # special case when predicting prot-prot interation: Z and R are not stored to save space
        # To avoid other problems in the code (we use data.R or data.Z to get the device right), we add dummy data here
        if not hasattr(data, "Z"):
            assert hasattr(data, "N_aa_chain1"), str(data)
            assert not hasattr(data, "R"), str(data)
            assert not hasattr(data, "N"), str(data)
            data.Z = torch.zeros_like(data.N_aa_chain1)
            data.R = torch.zeros_like(data.N_aa_chain1).type(floating_type)
            data.N = data.N_aa_chain1

        # initialization and data checking
        # It will only run once
        if self._compute_n_pl is None:
            self._compute_n_pl = not (hasattr(data, "N_p") and hasattr(data, "N_l"))
        if self._edge_precalculated is None:
            self._edge_precalculated = True
            for bond_type in self.bonding_types:
                if hasattr(data, f"{bond_type}_edge_index"):
                    continue
                self._edge_precalculated = False
                # XX_oneway edge will be computed along with XX edge
                bond_type = bond_type.split("_oneway")[0]
                if hasattr(data, f"{bond_type}_oneway_edge_index"):
                    self.bond_type_need_concat.add(bond_type)
                else:
                    self.bond_type_need_compute.add(bond_type)

        for key in self._drop:
            if hasattr(data, key):
                delattr(data, key)

        data = self.do_compute_edge(data)

        data.sample_id = idx
        data.atom_mol_batch = torch.zeros_like(data.Z)

        return data

    def do_compute_edge(self, data):
        if self._edge_precalculated:
            return data

        for bond_type in self.bond_type_need_concat:
            oneway_edge = getattr(data, f"{bond_type}_oneway_edge_index")
            oneway_dist = getattr(data, f"{bond_type}_oneway_dist")
            oneway_dist_mask = (oneway_dist <= self.cutoff_query[bond_type])
            oneway_edge = oneway_edge[:, oneway_dist_mask]
            oneway_dist = oneway_dist[oneway_dist_mask]

            concat_edge = torch.concat([oneway_edge, oneway_edge[[1, 0], :]], dim=-1)
            concat_dist = torch.concat([oneway_dist, oneway_dist], dim=0)
            setattr(data, f"{bond_type}_edge_index", concat_edge)
            setattr(data, f"{bond_type}_dist", concat_dist)

        query_success: bool = self.query_cache(data)
        if not query_success:
            self.compute_mol_bonds(data)
        return data

    def query_cache(self, data):
        if not self.cache_bonds: return False

        file_handle = data.FileHandle[0]
        if file_handle not in self.bond_cache_query:
            return False

        for key in self.bond_cache_query[file_handle]:
            setattr(data, key, self.bond_cache_query[file_handle][key])
        return True


    def compute_mol_bonds(self, data):
        if self._is_combined_pl: return
        pair_dist = pairwise_dist(data.R, data.R)
        for bond_type in self.bond_type_need_compute:
            if bond_type == "BN":
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
            elif bond_type == "L":
                edge_index = torch.nonzero(pair_dist>=self.cutoff_query[bond_type])
            else:
                assert bond_type in ["PL_min_dist_sep_oneway"], bond_type
                continue
            edge_index = edge_index.T
            setattr(data, f"{bond_type}_edge_index", edge_index)

            if not self.cache_bonds: continue
            file_handle = data.FileHandle[0]
            assert file_handle not in self.bond_cache_query, f"It should not be double-computed!"
            self.bond_cache_query[file_handle] = {f"{bond_type}_edge_index": edge_index}

    def drop_key(self, key):
        self._drop.add(key)

    @lazy_property
    def cutoff_query(self):
        query = {}
        # infer cutoff from config
        for bonding_type, cutoff in zip(self.cfg.model.bonding_type.split(), self.cfg.model.cutoffs.split()):
            try:
                if bonding_type not in query:
                    query[bonding_type] = float(cutoff)
                else:
                    assert query[bonding_type] == float(cutoff), f"{query}, {bonding_type}, {cutoff}"
            except ValueError:
                # ignore "None" cutoff which is not needed for preprocessing edges
                assert cutoff.lower() == "none"
        return query
