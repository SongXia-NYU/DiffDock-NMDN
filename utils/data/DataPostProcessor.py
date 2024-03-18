from utils.DataPrepareUtils import pairwise_dist
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
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        if cfg is None: return
        # it saves all bond types that are needed to be computed on the fly.
        self.bond_type_need_compute = set()
        self.bond_type_need_concat = set()

        self._drop = set()
        self.cut_protein = not cfg["no_cut_protein"] if cfg is not None else None
        self.proc_in_gpu = cfg["proc_in_gpu"] if cfg is not None else False

        self._is_combined_pl = None
        self._edge_precalculated = None
        self._compute_n_pl = None

        self._cutoff_query = None
        self._pl_cutoff = self.cfg["pl_cutoff"]
        self.request_voronoi: bool = ("LIGAND_Voronoi1" in self.bonding_types) or ("PROTEIN_Voronoi1" in self.bonding_types) or cfg["mdn_voronoi_edge"]
        self.cache_bonds: bool = cfg["cache_bonds"]
        if self.cache_bonds: self.bond_cache_query: Dict[str, Dict[str, torch.Tensor]] = {}

    @lazy_property
    def bonding_types(self) -> Set[str]:
        # a set containing all of the bond types needed for the run.
        # the processor will check if they are already provided or not. 
        # if not, they will be calculated on the fly.
        btype_list = self.cfg["bonding_type"].split()
        bonding_types = set([s for s in btype_list if s.lower()!="none"]) if self.cfg is not None else None
        # for backward compatibility
        if self.cfg["loss_metric"] == "mdn" and btype_list[-1].lower() == "none":
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

        # _is_combined_pl means if we combined both protein and ligand into a single 'molecule'
        if self._is_combined_pl is None:
            self._is_combined_pl = hasattr(data, "mol_type")

            if not self._edge_precalculated and self._is_combined_pl:
                # the mol_type of ligand atoms is 1
                n_l = (data.mol_type == 1).sum()
                for i in range(n_l):
                    assert data.mol_type[i] == 1, f"PL data: ligand must come first for correct edge calculation: {data.mol_type}"

        for key in self._drop:
            if hasattr(data, key):
                delattr(data, key)

        if self._is_combined_pl and self._compute_n_pl:
            data.N_p = (data.mol_type == 0).sum()
            data.N_l = (data.mol_type == 1).sum()

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
            self.compute_pl_bonds(data)
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

    def compute_pl_bonds(self, data):
        if not self._is_combined_pl: return

        if self.cache_bonds:
            # not implemented yet
            raise NotImplementedError
        N_l = data.N_l.item()
        if self.cut_protein:
            # cut the protein to pocket for backward compatibility
            data = self.do_cut_protein(data)

        protein_mask = (data.mol_type == 0)
        ligand_mask = (data.mol_type == 1)
        R_p = data.R[protein_mask, :]
        R_l = data.R[ligand_mask, :]

        for bond_type in self.bond_type_need_compute:
            if bond_type == "PROTEIN":
                pair_dist = pairwise_dist(R_p, R_p)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                # remove self-self interaction.
                # Note: do not use torch.nonzero(pair_dist > 0) to determine self-self interaction because of numerical instability
                non_self_edge = (edge_index[:, 0] != edge_index[:, 1])
                edge_index = edge_index[non_self_edge, :] + N_l
            elif bond_type == "LIGAND":
                pair_dist = pairwise_dist(R_l, R_l)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                non_self_edge = (edge_index[:, 0] != edge_index[:, 1])
                edge_index = edge_index[non_self_edge, :]
            elif bond_type == "PL":
                pair_dist = pairwise_dist(R_p, R_l)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                pl_dist = pair_dist[edge_index[:, 0], edge_index[:, 1]]
                edge_index[:, 0] += N_l

                # the one-way edges are used by MDN layers
                setattr(data, f"{bond_type}_oneway_dist", pl_dist)
                setattr(data, f"{bond_type}_oneway_edge_index", edge_index.T)
                # both source to target and target to source flow
                # Took me hours to find this bug
                edge_index = torch.concat([edge_index, edge_index[:, [1, 0]]])
            elif bond_type == "P_BETA_oneway":
                N_p = data.N_p.item()
                # the mol_type of beta atoms is 2
                beta_mask = data.mol_type == 2
                R_beta = data.R[beta_mask, :]
                pair_dist = pairwise_dist(R_p, R_beta)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                # since the data is organized as ligand-protein-beta_atoms, 
                # the corresponding indices should be shifted by such numbers below.
                edge_index[:, 0] += N_l
                edge_index[:, 1] += N_l + N_p
                # unlike in PL edge, the P_BETA edge is one-way, i.e., from protein to beta atoms
            elif bond_type == "BETA_L_oneway":
                N_p = data.N_p.item()
                # the mol_type of beta atoms is 2
                beta_mask = data.mol_type == 2
                R_beta = data.R[beta_mask, :]
                pair_dist = pairwise_dist(R_beta, R_l)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                this_dist = pair_dist[edge_index[:, 0], edge_index[:, 1]]
                # since the data is organized as ligand-protein-beta_atoms, 
                # the corresponding indices should be shifted by such numbers below.
                edge_index[:, 0] += N_l + N_p
                # the one-way edges are used by MDN layers
                setattr(data, f"{bond_type}_dist", this_dist)
            elif bond_type == "BETA":
                N_p = data.N_p.item()
                N_l = data.N_l.item()
                # the mol_type of beta atoms is 2
                beta_mask = data.mol_type == 2
                R_beta = data.R[beta_mask, :]

                pair_dist = pairwise_dist(R_beta, R_beta)
                edge_index = torch.nonzero(pair_dist<self.cutoff_query[bond_type])
                # remove self-self interaction.
                # Note: do not use torch.nonzero(pair_dist > 0) to determine self-self interaction because of numerical instability
                non_self_edge = (edge_index[:, 0] != edge_index[:, 1])
                edge_index = edge_index[non_self_edge, :] + N_l + N_p
            else:
                assert bond_type.endswith("_Voronoi1"), bond_type
                continue
            setattr(data, f"{bond_type}_edge_index", edge_index.T)

    def drop_key(self, key):
        self._drop.add(key)

    @lazy_property
    def cutoff_query(self):
        query = {}
        # infer cutoff from config
        for bonding_type, cutoff in zip(self.cfg["bonding_type"].split(), self.cfg["cutoffs"].split()):
            try:
                if bonding_type not in query:
                    query[bonding_type] = float(cutoff)
                else:
                    assert query[bonding_type] == float(cutoff), f"{query}, {bonding_type}, {cutoff}"
            except ValueError:
                # ignore "None" cutoff which is not needed for preprocessing edges
                assert cutoff.lower() == "none"
        # specially for MDN loss, the PL cutoff is needed
        if self.cfg["loss_metric"].startswith("mdn"):
            pl_cutoff = max(self.cfg["mdn_threshold_train"], self.cfg["mdn_threshold_eval"])
            if "PL" in query:
                # TODO: this is due to code logic
                assert query["PL"] >= pl_cutoff, "PL cutoff during message passing must be higher than MDN cutoff"
            else:
                query["PL"] = pl_cutoff

        return query

    @property
    def pl_cutoff(self):
        if self._pl_cutoff is None:
            if self.cfg["add_sqf"] is not None:
                # backward compatibility
                self._pl_cutoff = float(self.cfg["add_sqf"][0].split("$")[-2])
        return self._pl_cutoff
    
    def do_cut_protein(self, data:MyData):
        raise ValueError("Cutting protein is discontinued.")
        N_l = data.N_l.item()
        R_p = data.R[N_l:, :]
        R_l = data.R[:N_l, :]
        pair_dist = pairwise_dist(R_l, R_p)
        selected_protein_idx = torch.nonzero(pair_dist<self.pl_cutoff)[:, 1]
        selected_protein_idx = torch.unique(selected_protein_idx)
        if self.request_voronoi:
            # Here we do some magic trick to map the pre-calculated full-protein PL_Voronoi edge index to the cut-protein edge index
            # we assume the pre-calculated full-protein PL_Voronoi edge index is ligand -> protein
            protein_mapper = torch.zeros_like(data.Z).fill_(-1)
            for idx_after, idx_sel in enumerate(selected_protein_idx.cpu().detach().numpy()):
                protein_mapper[N_l+idx_sel] = idx_after + N_l
            protein_updated_idx = protein_mapper[data.PL_Voronoi1_edge_index[1, :]]
            protein_updated_idx_masker = (protein_updated_idx != -1)
            protein_updated_idx = protein_updated_idx[protein_updated_idx_masker]
            data.PL_Voronoi1_edge_index = torch.cat([data.PL_Voronoi1_edge_index[0, protein_updated_idx_masker].view(1, -1), protein_updated_idx.view(1, -1)], dim=0)

        R_p = R_p[selected_protein_idx]
        data.R = torch.concat([R_l, R_p], dim=0)
        data.Z = torch.concat([data.Z[:N_l], data.Z[N_l:][selected_protein_idx]], dim=0)
        data.mol_type = torch.concat([data.mol_type[:N_l], data.mol_type[N_l:][selected_protein_idx]], dim=0)
        data.N = data.Z.shape[0]
        data.N_p = R_p.shape[0]
        return data