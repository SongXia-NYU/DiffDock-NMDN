import argparse
from copy import deepcopy
import os
from typing import List

import pandas as pd
import psutil

import torch
import os.path as osp
import numpy as np
import ase
import json
from dscribe.descriptors import ACSF
from torch_scatter import scatter_add
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from tempfile import TemporaryDirectory
from prody.proteins import writePDB

import tqdm
from torch_geometric.data import Data, HeteroData
from multiprocessing import Pool

from geometry_processors.DataPrepareUtils import my_pre_transform, set_force_cpu, voronoi_edge_index, pairwise_dist
from geometry_processors.pl_dataset.ConfReader import PDBHeteroReader, PDBReader, SDFReader, MolReader, PDBLegacyReader, CGMartiniPDBReader, Mol2Reader, BetaAtomPDBReader
from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory
from geometry_processors.pl_dataset.csv2input_list import csv2input_list, MPInfo
from geometry_processors.pl_dataset.prot_utils import pl_min_dist_matrix, pp_min_dist_matrix_vec_mem


ACSF_SPECIES = ['I', 'Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'P', 'S', 'Ca', 'Zn']
def _symbol2number(symbol):
    atom = ase.Atom(symbol)
    return atom.number
ACSF_MASKER = torch.as_tensor([_symbol2number(sym) for sym in ACSF_SPECIES]).view(1, -1)


class PLProcessor:
    def __init__(self, info: MPInfo, legacy=False, protein_reader_args=None, ligand_reader_args=None, neutral=False,
                 martini=False, compute_edge=True, acsf=False):
        self.protein_reader_args = protein_reader_args
        self.ligand_reader_args = ligand_reader_args
        self.info = info
        self.acsf = acsf
        # if compute_edge, edge_index will be pre-computed and saved into memory
        # in this way, only the protein atoms within the pocket will remain, other atoms/beads will lost.
        self.compute_edge = compute_edge

        self._ligand_reader = None
        self._protein_reader = None
        self._ligand_file = None
        self._protein_file = None
        self.conf_reader_factory = ConfReaderFactory(info, protein_reader_args, ligand_reader_args, legacy=legacy, martini=martini, neutral=neutral)

    @property
    def ligand_reader(self):
        if self._ligand_reader is None:
            self._ligand_reader, self._ligand_file = self.conf_reader_factory.get_lig_reader()
        return self._ligand_reader

    @property
    def ligand_file(self):
        if self._ligand_file is None:
            __ = self.ligand_reader
        return self._ligand_file

    @property
    def protein_reader(self):
        if self._protein_reader is None:
            self._protein_reader, self._protein_file = self.conf_reader_factory.get_prot_reader()
        return self._protein_reader

    @property
    def protein_file(self):
        if self._protein_file is None:
            __ = self.protein_reader
        return self._protein_file

    def process_single_entry(self, save=True):
        if save and osp.exists(self.info.pyg_name):
            # skip the file if it exists and can be successfully loaded by PyTorch
            try:
                torch.load(self.info.pyg_name)
                print(f"Skipping: {self.info.pyg_name}")
                return
            except Exception as e:
                os.remove(self.info.pyg_name)
        
        ligand_dict = self.ligand_reader.get_basic_dict()
        # calculate ligand edge index based on cutoff 10
        R_l = ligand_dict["R"]
        pair_dist = pairwise_dist(R_l, R_l)
        edge_index = torch.nonzero(pair_dist<10.)
        non_self_edge = (edge_index[:, 0] != edge_index[:, 1])
        edge_index = edge_index[non_self_edge, :].T
        ligand_dict["LIGAND_edge_index"] = edge_index

        if self.info.protein_pdb is None:
            system_dict = ligand_dict
            system_dict["N_l"] = system_dict["N"]
            system_dict["N_p"] = torch.zeros_like(system_dict["N"])
            system_dict["ligand_file"] = [self.ligand_file]
            system_dict["protein_file"] = [""]
            system_dict["mol_type"] = torch.ones(system_dict["N"]).long()
            system_dict.update(self.info.labels)
            return Data(**system_dict)
        
        n_atom_ligand = ligand_dict["N"].item()
        protein_dict = self.protein_reader.get_basic_dict()

        # compute edge based on distance cutoff. More efficient way is possible in the sPhyset script (i.e. do not pre-compute edge here)
        # TODO: remove the compute_edge script here
        if self.compute_edge:
            # calculate intra-molecular interaction: ligand
            d_ligand = Data(**ligand_dict)
            d_ligand = my_pre_transform(d_ligand, edge_version="cutoff", do_sort_edge=False, cal_efg=False,
                                        cutoff=self.info.cutoff_ligand, boundary_factor=100., use_center=True, mol=None,
                                        cal_3body_term=False, bond_atom_sep=False, record_long_range=False,
                                        extended_bond=False)
            ligand_bn_edge_index = d_ligand.BN_edge_index
            del d_ligand

            ligand_r = ligand_dict["R"]
            ligand_index_selector = torch.arange(n_atom_ligand).long()

            # calculate inter-molecular interaction: ligand and protein
            n_atom_protein = protein_dict["N"].item()
            protein_idx = n_atom_ligand
            protein_r_cutoff = []
            protein_z_cutoff = []
            inter_edge_index = []
            for i in range(n_atom_protein):
                this_r = protein_dict["R"][i, :].view(1, -1)
                this_z = protein_dict["Z"][i]
                dist_sq: torch.Tensor = ((ligand_r - this_r) ** 2).sum(dim=-1)
                mask: torch.Tensor = (dist_sq < self.info.cutoff_pl ** 2)

                if mask.sum() > 0:
                    edge_ligand = ligand_index_selector[mask].view(1, -1)
                    edge_protein = torch.zeros_like(edge_ligand).fill_(protein_idx).view(1, -1)
                    edge_index1 = torch.concat([edge_ligand, edge_protein], dim=0)
                    edge_index2 = torch.concat([edge_protein, edge_ligand], dim=0)
                    edge_index = torch.concat([edge_index1, edge_index2], dim=-1)

                    inter_edge_index.append(edge_index)
                    protein_r_cutoff.append(this_r)
                    protein_z_cutoff.append(this_z.item())
                    protein_idx += 1
            protein_context = len(protein_r_cutoff) > 0
            if protein_context:
                protein_r_cutoff = torch.concat(protein_r_cutoff, dim=0)
                protein_z_cutoff = torch.as_tensor(protein_z_cutoff).long()
                inter_edge_index = torch.concat(inter_edge_index, dim=-1)
            else:
                # no context protein
                protein_r_cutoff = torch.zeros((0, 3), dtype=ligand_dict["R"].dtype)
                protein_z_cutoff = torch.zeros(0).long()
                inter_edge_index = torch.zeros((2, 0)).long()

            # calculate intra-molecular interaction: protein
            protein_dict = {
                "R": protein_r_cutoff,
                "Z": protein_z_cutoff,
                "N": torch.as_tensor(protein_z_cutoff.shape[0]).view(-1)
            }
            if protein_context:
                d_protein = Data(**protein_dict)
                d_protein = my_pre_transform(d_protein, edge_version="cutoff", do_sort_edge=False, cal_efg=False,
                                            cutoff=self.info.cutoff_protein, boundary_factor=100., use_center=True,
                                            mol=None,
                                            cal_3body_term=False, bond_atom_sep=False, record_long_range=False,
                                            extended_bond=False)
                protein_bn_edge_index = d_protein.BN_edge_index + n_atom_ligand
                del d_protein
            else:
                protein_bn_edge_index = torch.zeros((2, 0)).long()

        system_dict = {
            "R": torch.concat([ligand_dict["R"], protein_dict["R"]], dim=0),
            "Z": torch.concat([ligand_dict["Z"], protein_dict["Z"]], dim=0),
            "N": ligand_dict["N"] + protein_dict["N"],
            "N_p": protein_dict["N"],
            "N_l": ligand_dict["N"],
            "ligand_file": [self.ligand_file],
            "protein_file": [self.protein_file],
            "LIGAND_edge_index": ligand_dict["LIGAND_edge_index"]
        }
        # 0: protein; 1: ligand. First ligand then protein
        system_dict["mol_type"] = torch.zeros(system_dict["N"]).long()
        system_dict["mol_type"][:n_atom_ligand] = 1

        if self.acsf:
            # un-supported atoms are removed
            type_masker = torch.any(system_dict["Z"].view(-1, 1) == ACSF_MASKER, dim=-1)
            for key in ["R", "Z", "mol_type"]:
                system_dict[key] = system_dict[key][type_masker]
            system_dict["N"] = system_dict["Z"].shape[0]
            system_dict["N_l"] = system_dict["mol_type"].sum()
            system_dict["N_p"] = system_dict["N"] - system_dict["N_l"]


            atoms = ase.Atoms(numbers=system_dict["Z"].numpy().tolist(), positions=system_dict["R"].numpy())
            acsf = ACSF(species=ACSF_SPECIES,
                        rcut=6.0, g2_params=[[1, 1], [1, 2], [1, 3]],
                        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]])
            acsf_features = acsf.create(atoms, positions=np.arange(system_dict["N_l"]))
            system_dict["acsf_features"] = torch.as_tensor(acsf_features).float()

        system_dict = self.infuse_labels(system_dict)

        if self.compute_edge:
            system_dict["LIGAND_edge_index"] = ligand_bn_edge_index
            system_dict["PROTEIN_edge_index"] = protein_bn_edge_index
            system_dict["PL_edge_index"] = inter_edge_index
            system_dict["num_LIGAND_edge"] = torch.as_tensor(ligand_bn_edge_index.shape[-1])
            system_dict["num_PROTEIN_edge"] = torch.as_tensor(protein_bn_edge_index.shape[-1])
            system_dict["num_PL_edge"] = torch.as_tensor(inter_edge_index.shape[-1])

        d_system = Data(**system_dict)
        if save:
            os.makedirs(osp.dirname(self.info.pyg_name), exist_ok=True)
            torch.save(d_system, self.info.pyg_name)
            return
        else:
            return d_system
        
    def infuse_labels(self, system_dict):
        for key in self.info.labels:
            val = self.info.labels[key]
            if isinstance(val, (int, float, bool)):
                system_dict[key] = torch.as_tensor(val).view(-1)
            else:
                assert isinstance(val, str), f"unsopported type:{val}, {val.__class__}"
                system_dict[key] = val
        return system_dict


class PLVoronoiEdgeCalculator(PLProcessor):
    def __init__(self, info: MPInfo, protein_reader_args=None, ligand_reader_args=None, martini=False):
        super().__init__(info, legacy=False, protein_reader_args=protein_reader_args, ligand_reader_args=ligand_reader_args, 
            neutral=False, martini=martini, compute_edge=False, acsf=False)

    @staticmethod
    def pl_voronoi_edge(R_concat, N_l):
        """
        Calculate Voronoi edges given a concatnated R (assuming the first N_l atoms are ligand atoms) 
        and number of ligand atoms N_l
        """
        voronoi_edge_all = voronoi_edge_index(R_concat, 2.0, True)

        # ligand atoms have indices between 0 and N_l-1. 
        # protein beads/atoms have indices between N_l and N_l+N_p-1.
        mask_first_is_lig = (voronoi_edge_all[0, :] < N_l)
        mask_second_is_lig = (voronoi_edge_all[1, :] < N_l)
        mask_ll = (mask_first_is_lig & mask_second_is_lig)
        mask_pp = ((~mask_first_is_lig) & (~mask_second_is_lig))
        mask_pl = ~ (mask_pp | mask_ll)

        ll_edge = voronoi_edge_all[:, mask_ll]
        pp_edge = voronoi_edge_all[:, mask_pp]
        pl_edge = voronoi_edge_all[:, mask_pl]
        return ll_edge, pp_edge, pl_edge
    
    def process_single_entry(self, save=False):
        d = super().process_single_entry(save=False)
        N_l = d.N_l
        ll_edge, pp_edge, pl_edge = self.pl_voronoi_edge(d.R, N_l)
        setattr(d, "LIGAND_Voronoi1_edge_index", ll_edge)
        if self.info.discard_p_info:
            d.R = d.R[:N_l]
            d.Z = d.Z[:N_l]
            d.mol_type = d.mol_type[:N_l]
            setattr(d, "PROTEIN_Voronoi1_edge_index", torch.empty(2, 0))
        else:
            setattr(d, "PROTEIN_Voronoi1_edge_index", pp_edge)

        # sort edge so that edge_index[0, :] is ligand and edge_index[1, :] is protein
        v_edge_sorted = []
        for i in range(pl_edge.shape[-1]):
            this_edge = pl_edge[:, [i]]
            if this_edge[0, 0] < N_l:
                assert this_edge[1, 0] >= N_l
            else:
                assert this_edge[1, 0] < N_l, this_edge
                this_edge = this_edge[[1, 0], :]
            v_edge_sorted.append(this_edge)
        v_edge_sorted = torch.cat(v_edge_sorted, dim=-1)
        pl_edge = v_edge_sorted
        setattr(d, "PL_Voronoi1_edge_index", pl_edge)

        if save:
            os.makedirs(osp.dirname(self.info.pyg_name), exist_ok=True)
            torch.save(d, self.info.pyg_name)
            return self.info.pyg_name

        return d


class PLMultiResProcessor(PLVoronoiEdgeCalculator):
    """
    Process dataset into Pytorch Geometric format with multi-resolution information, 
    including: Martini Bead information and AA center of mass and some descriptors inspired from RTMScore.
    """
    def __init__(self, info: MPInfo, protein_reader_args=None, ligand_reader_args=None, error_dir=None):
        super().__init__(info, protein_reader_args, ligand_reader_args, martini=True)

        self._protein_atom_reader = None
        self.error_dir = error_dir

    @property
    def protein_atom_reader(self):
        if self._protein_atom_reader is None:
            self._protein_atom_reader = PDBReader(self.info.protein_atom_pdb, stdaa=True)
        return self._protein_atom_reader

    def save_error(self):
        assert self.error_dir
        pdb = osp.basename(self.info.protein_pdb).split(".")[0].split("_")[0]
        with open(osp.join(self.error_dir, f"{pdb}.json"), "w") as f:
            json.dump(self.get_error_info(), f, indent=2)

    def get_error_info(self):
        info = {"num_aa_martini": self.protein_reader.prody_parser.numResidues(),
                "num_aa_atom": self.protein_atom_reader.prody_parser.numResidues()}
        martini_seq = "".join([res.getSequence()[0] for res in self.protein_reader.prody_parser.iterResidues()])
        atom_seq = "".join([res.getSequence()[0] for res in self.protein_atom_reader.prody_parser.iterResidues()])
        info["martini_seq"] = martini_seq
        info["atom_seq"] = atom_seq
        return info
    
    def process_single_entry(self, save=False):
        # the processed data with Martini and voronoi edge information
        d_raw = super().process_single_entry(save)

        # mapping Martini Beads to AA ids
        martini_aa_batch = self.protein_reader.prody_parser.getResindices()
        d_raw.martini_aa_batch = torch.as_tensor(martini_aa_batch).long()

        # Calculate the center of mass
        prot_atom_dict = self.protein_atom_reader.get_basic_dict()
        R_prot_atom = prot_atom_dict["R"]
        atom_masses = torch.as_tensor(self.protein_atom_reader.prody_parser.getMasses()).view(-1, 1)
        R_prot_atom_massed = R_prot_atom * atom_masses
        atom_aa_batch = torch.as_tensor(self.protein_atom_reader.prody_parser.getResindices()).long().view(-1)
        R_masses_summed = scatter_add(R_prot_atom_massed, atom_aa_batch, dim=0)
        mass_summed = scatter_add(atom_masses, atom_aa_batch, dim=0)
        center_of_mass = R_masses_summed / mass_summed
        d_raw.R_aa = center_of_mass
        d_raw.N_p_aa = center_of_mass.shape[0]
        # Calculate AA-level Voronoi edges
        R_l_aa_concat = torch.concat([d_raw.R[:d_raw.N_l, :], center_of_mass], dim=0)
        ll_edge, pp_edge, pl_edge = self.pl_voronoi_edge(R_l_aa_concat, d_raw.N_l)
        setattr(d_raw, "PROTEIN_AA_Voronoi1_edge_index", pp_edge)
        setattr(d_raw, "PL_AA_Voronoi1_edge_index", pl_edge)

        # Calculate extra AA-level properties inspired by RTMScore
        with TemporaryDirectory() as temp_d:
            temp_pdb = osp.join(temp_d, "temp.pdb")
            writePDB(temp_pdb, self.protein_atom_reader.prody_parser)
            res_features = self.pdb2aa_features(temp_pdb)
        setattr(d_raw, "feats_prot_aa", res_features)
        assert d_raw.N_p_aa == res_features.shape[0], f"dimensions are not consistent: {d_raw.N_p_aa} != {res_features.shape[0]}"
        assert d_raw.martini_aa_batch.max().item() == d_raw.N_p_aa-1, f"number of AAs does not match: {d_raw.martini_aa_batch.max().item()}, {d_raw.N_p_aa-1}"
        return d_raw

    @staticmethod
    def pdb2aa_features(pdb_file):
        u = mda.Universe(pdb_file)
        res_feats = np.array([PLMultiResProcessor.calc_res_features(res) for res in u.residues])
        res_feats = torch.as_tensor(res_feats)
        return res_feats
    
    @staticmethod
    def calc_res_features(res):
        # self distances: min, max, alpha carbon to oxygen, oxygen to nitrogen, nitrogen to carbon. 5 dimensions.
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        try:
            self_dists = [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
        except ValueError:
            # It happens for water molecules
            self_dists = [0., 0., 0., 0., 0.]
        except IndexError:
            # It happens when the residue has missing heavy atoms
            self_dists = [0., 0., 0., 0., 0.]
        # dihediral_angles. 4 dimensions
        try:
            if res.phi_selection() is not None:
                phi = res.phi_selection().dihedral.value()
            else:
                phi = 0
            if res.psi_selection() is not None:
                psi = res.psi_selection().dihedral.value()
            else:
                psi = 0 
            if res.omega_selection() is not None:
                omega = res.omega_selection().dihedral.value()
            else:
                omega = 0
            if res.chi1_selection() is not None:
                chi1 = res.chi1_selection().dihedral.value()
            else:
                chi1 = 0
            dihediral_angles = [phi*0.01, psi*0.01, omega*0.01, chi1*0.01]
        except mda.exceptions.SelectionError:
            dihediral_angles = [0., 0., 0., 0.]

        # Output is the concat of self_dist and dihediral angles, 9 dimensions in total
        self_dists.extend(dihediral_angles)
        return self_dists

class PLVoronoiBetaAtomProcessor(PLVoronoiEdgeCalculator):
    """
    Process data and then add Beta Atoms from AlphaSpace2 program. Calculate the Voronoi Edge between Beta Atoms and Protein.
    """
    def __init__(self, info: MPInfo, protein_reader_args=None, ligand_reader_args=None, martini=False):
        super().__init__(info, protein_reader_args, ligand_reader_args, martini)
        self.beta_atom_reader = BetaAtomPDBReader(info.beta_atom_pdb)

    def process_single_entry(self, save=False):
        d_raw = super().process_single_entry(save)
        beta_atom_dict = self.beta_atom_reader.get_basic_dict()
        d_raw.N_beta = beta_atom_dict["N"]
        d_raw.N = d_raw.N + beta_atom_dict["N"]
        d_raw.R = torch.concat([d_raw.R, beta_atom_dict["R"]], dim=0)
        d_raw.Z = torch.concat([d_raw.Z, beta_atom_dict["Z"]], dim=0)
        # the mol type ID of beta atoms is 2
        beta_mol_type = torch.zeros_like(beta_atom_dict["Z"]).fill_(2)
        d_raw.mol_type = torch.concat([d_raw.mol_type, beta_mol_type], dim=0)
        return d_raw


class PLPaddingStyleProcessor(PLProcessor):
    """
    Process protein in a padding style: the protein_R is [N_aa, N_max_atom_per_aa, 3] 
    and protein_Z is [N_aa, N_max_atom_per_aa]
    """
    def __init__(self, info: MPInfo, legacy=False, protein_reader_args=None, ligand_reader_args=None):
        super().__init__(info, legacy, protein_reader_args, ligand_reader_args)
    
    def process_single_entry(self, return_dict=False):
        ligand_dict = self.ligand_reader.get_basic_dict()
        protein_pad_dict = self.protein_reader.get_padding_style_dict()

        out_dict = ligand_dict
        out_dict["ligand_file"] = [self.ligand_file]

        out_dict["protein_file"] = [self.protein_file]
        out_dict["R_prot_pad"] = torch.as_tensor(protein_pad_dict["R"])
        out_dict["Z_prot_pad"] = torch.as_tensor(protein_pad_dict["Z"])
        out_dict["N_prot"] = torch.as_tensor(protein_pad_dict["N"])
        if self.info.discard_p_info:
            out_dict["protein_file"] = [""]
            out_dict["R_prot_pad"] = torch.zeros((0, protein_pad_dict["R"].shape[1], 3))
            out_dict["Z_prot_pad"] = torch.zeros((0, protein_pad_dict["R"].shape[1]))
            out_dict["N_prot"] = torch.zeros((0, ))

        pl_min_dist = pl_min_dist_matrix(ligand_dict["R"], protein_pad_dict["R"]).squeeze(0)

        edge_index = torch.nonzero(pl_min_dist<=self.info.cutoff_pl)
        pl_dist = pl_min_dist[edge_index[:, 0], edge_index[:, 1]]

        bond_type = "PL_min_dist_sep"
        # the one-way edges are used by MDN layers
        out_dict[f"{bond_type}_oneway_dist"] = pl_dist
        out_dict[f"{bond_type}_oneway_edge_index"] = edge_index.T
        out_dict = self.infuse_labels(out_dict)
        if return_dict:
            # [n_residue, ]
            out_dict["pl_min_dist_by_residue"] = pl_min_dist.min(dim=0)[0]
            return out_dict
        return Data(**out_dict)

class PLHeteroProcessor(PLPaddingStyleProcessor):
    """
    Process protein-ligand file and return torch_geometric HeteroGraph.
    The output graph contains protein-ligand, water-ligand and ion-ligand interaction.
    """
    def __init__(self, info: MPInfo, legacy=False, protein_reader_args=None, ligand_reader_args=None):
        super().__init__(info, legacy, protein_reader_args, ligand_reader_args)

    @property
    def protein_reader(self):
        if self._protein_reader is not None:
            return self._protein_reader
        
        prot_file: str = self.info.protein_pdb
        reader = PDBHeteroReader(prot_file)
        self._protein_reader = reader
        self._protein_file = prot_file
        return self._protein_reader

    def process_single_entry(self) -> HeteroData:
        pl_dict = super().process_single_entry(True)

        # protein-ligand interaction
        prot_dict = {"R": torch.zeros((0, pl_dict["R_prot_pad"].shape[1], 3)),
                     "Z": torch.zeros((0, pl_dict["Z_prot_pad"].shape[1])),
                     "N": pl_dict["N_prot"]}
        lig_dict = {"R": pl_dict["R"], "Z": pl_dict["Z"], "N": pl_dict["N"]}
        out_dict = {"protein": prot_dict, "ligand": lig_dict}
        pl_inter_dict = {"min_dist_edge_index": pl_dict["PL_min_dist_sep_oneway_edge_index"],
                         "min_dist": pl_dict["PL_min_dist_sep_oneway_dist"]}
        out_dict["ligand", "interaction", "protein"] = pl_inter_dict

        # protein-protein interaction
        if False:
            pp_min_dist = pp_min_dist_matrix_vec_mem(pl_dict["R_prot_pad"], pl_dict["R_prot_pad"])
            edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_pl)
            # only retain one-way edge to save space. The two-way edge will be computed on-the-fly
            edge_index_mask = edge_index[:, 0] < edge_index[:, 1]
            edge_index = edge_index[edge_index_mask, :]
            # for extremely large proteins, the pp edge will be quadratically large
            # to avoid this problem, only protein residue within 30A is considered for pp interaction
            edge_index_mask = (pl_dict["pl_min_dist_by_residue"][edge_index[:, 0]]) < 30.
            edge_index = edge_index[edge_index_mask, :]
            edge_index_mask = (pl_dict["pl_min_dist_by_residue"][edge_index[:, 1]]) < 30.
            edge_index = edge_index[edge_index_mask, :]

            pp_min_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]
            pp_dict = {"min_dist_edge_index": edge_index.T, "min_dist": pp_min_dist}
            out_dict["protein", "interaction", "protein"] = pp_dict

        # ion-ligand interaction
        hetero_dict = self.protein_reader.parse_hetero_graph()
        out_dict["ion"] = hetero_dict["ion"]
        lig_ion_min_dist = pl_min_dist_matrix(lig_dict["R"], hetero_dict["ion"]["R"].view(-1, 1, 3)).squeeze(0)
        edge_index = torch.nonzero(lig_ion_min_dist<=self.info.cutoff_pl)
        lig_ion_min_dist = lig_ion_min_dist[edge_index[:, 0], edge_index[:, 1]]
        lig_ion_dict = {"min_dist_edge_index": edge_index.T, "min_dist": lig_ion_min_dist}
        out_dict["ligand", "interaction", "ion"] = lig_ion_dict

        # protein-ion interaction
        if hetero_dict["ion"]["R"].shape[0] > 0:
            prot_ion_min_dist = pl_min_dist_matrix(hetero_dict["ion"]["R"].view(-1, 3),
                                                pl_dict["R_prot_pad"]).squeeze(0)
            edge_index = torch.nonzero(prot_ion_min_dist<=self.info.cutoff_pl)
            prot_ion_min_dist = prot_ion_min_dist[edge_index[:, 0], edge_index[:, 1]]
            prot_ion_dict = {"min_dist_edge_index": edge_index.T, "min_dist": prot_ion_min_dist}
            out_dict["ion", "interaction", "protein"] = prot_ion_dict
        else:
            out_dict["ion", "interaction", "protein"] = deepcopy(lig_ion_dict)

        # water-ligand interaction
        out_dict["water"] = hetero_dict["water"]
        lig_water_min_dist = pl_min_dist_matrix(lig_dict["R"], hetero_dict["water"]["R"].view(-1, 3, 3)).squeeze(0)
        edge_index = torch.nonzero(lig_water_min_dist<=self.info.cutoff_pl)
        lig_water_min_dist = lig_water_min_dist[edge_index[:, 0], edge_index[:, 1]]
        lig_water_dict = {"min_dist_edge_index": edge_index.T, "min_dist": lig_water_min_dist}
        out_dict["ligand", "interaction", "water"] = lig_water_dict

        # misc info
        out_dict["protein_file"] = self.protein_file
        out_dict["ligand_file"] = self.ligand_file
        out_dict.update(self.info.labels)

        return HeteroData(out_dict)
    
class PLMinDistImplicitProcessor(PLPaddingStyleProcessor):
    """
    Calculate pairwise min distance for protein residue-residue interaction and residue-atom interaction.
    """
    def __init__(self, info: MPInfo, legacy=False, protein_reader_args=None, ligand_reader_args=None, cal_pp=True):
        super().__init__(info, legacy, protein_reader_args, ligand_reader_args)
        self.cal_pp = cal_pp
    
    def process_single_entry(self):
        out_dict = super().process_single_entry(return_dict=True)

        protein_pad_dict = self.protein_reader.get_padding_style_dict()
        # the protein atom coordinates are deleted to save space
        out_dict["R_prot_pad"] = torch.zeros((0, protein_pad_dict["R"].shape[1], 3))
        out_dict["Z_prot_pad"] = torch.zeros((0, protein_pad_dict["R"].shape[1]))
        if not self.cal_pp:
            return Data(**out_dict)

        # calculate protein residue-residue min distance
        pp_min_dist = pp_min_dist_matrix_vec_mem(protein_pad_dict["R"])
        pp_min_dist = torch.as_tensor(pp_min_dist)
        edge_index = torch.nonzero(pp_min_dist<=self.info.cutoff_protein)
        # remove self-self and reverse interaction
        edge_index_mask = edge_index[:, 0] < edge_index[:, 1]
        edge_index = edge_index[edge_index_mask, :]
        pp_dist = pp_min_dist[edge_index[:, 0], edge_index[:, 1]]

        bond_type = "PP_min_dist"
        # the one-way edges are used by MDN layers
        out_dict[f"{bond_type}_oneway_dist"] = pp_dist
        out_dict[f"{bond_type}_oneway_edge_index"] = edge_index.T
        out_dict = self.infuse_labels(out_dict)

        return Data(**out_dict)


class ProteinProcessor(PLProcessor):
    def __init__(self, info: MPInfo, protein_reader_args=None, martini=False):
        super().__init__(info, False, protein_reader_args, None, False, martini)
        set_force_cpu()

    def process_single_entry(self, save=True, precompute_edge=True, precompute_voronoi=False):
        if save and osp.exists(self.info.pyg_name):
            print(f"Skipping: {self.info.pyg_name}")
            return

        prot_dict = self.protein_reader.get_basic_dict()
        for key in self.info.labels:
            val = self.info.labels[key]
            if isinstance(val, (int, float, bool)):
                prot_dict[key] = torch.as_tensor(val).view(-1)
            else:
                assert isinstance(val, str), f"unsopported type:{val}, {val.__class__}"
                prot_dict[key] = val
        d_prot = Data(**prot_dict)
        if precompute_edge:
            d_prot = my_pre_transform(d_prot, edge_version="cutoff", do_sort_edge=False, cal_efg=False,
                                    cutoff=self.info.cutoff_protein, boundary_factor=100., use_center=True, mol=None,
                                    cal_3body_term=False, bond_atom_sep=False, record_long_range=False,
                                    extended_bond=False)
        if precompute_voronoi:
            from scipy.spatial import QhullError
            try:
                voronoi_edge = voronoi_edge_index(prot_dict["R"], 2.0, True)
            except QhullError:
                # it happens when only one atom/bead is present in the structure
                voronoi_edge = torch.empty(2, 0)
            setattr(d_prot, "PROTEIN_Voronoi1_edge_index", voronoi_edge)
        if save:
            os.makedirs(osp.dirname(self.info.pyg_name), exist_ok=True)
            torch.save(d_prot, self.info.pyg_name)
            return
        else:
            return d_prot


def single_wrapper(info: MPInfo):
    try:
        processor = PLProcessor(info)
        processor.process_single_entry()
        return pd.DataFrame()
    except Exception as e:
        error_df = pd.DataFrame({"protein": [osp.basename(info.protein_pdb)],
                                 "ligand": [osp.basename(info.ligand_sdf)],
                                 "msg": [f"{e}".split("\n")[0].replace(",", ".")]})
        return error_df


def all2single_pygs(csvs, root, debug=False):
    for csv in csvs:
        sub_name = osp.basename(csv).split(".")[0]
        info_list: List[MPInfo] = csv2input_list(csv, root)
        if debug:
            info_list = info_list[:10]
        with Pool(psutil.cpu_count()) as p:
            map_iter = tqdm.tqdm(p.imap_unordered(single_wrapper, info_list), total=len(info_list))
            errors = [e for e in map_iter]
        errors = pd.concat(errors)
        os.makedirs(osp.join(root, "errors"), exist_ok=True)
        errors.to_csv(osp.join(root, "errors", f"{sub_name}-errors.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", type=str, nargs='+')
    parser.add_argument("--root")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args = vars(args)

    set_force_cpu()
    all2single_pygs(**args)


def main_test():
    csv = "../../data/PDBbind_v2020/csv/CASF-2016.csv"
    root = "../../data/PDBbind_v2020"
    info_list: List[MPInfo] = csv2input_list(csv, root)
    for info in info_list:
        single_wrapper(info)

def main_processor():
    info = MPInfo(protein_pdb="/vast/sx801/PDBbind_v2020/structure_martini/pdb/3ce0_protein.martini.pdb", 
        ligand_sdf="/vast/sx801/PDBbind_v2020/structure_polarH/train_set_sdf/3ce0_ligand.polar.sdf",
        protein_atom_pdb="/vast/sx801/PDBbind_v2020/structure_polarH/train_set_protein/3ce0_protein.polar.pdb")
    processor = PLMultiResProcessor(info=info)
    d = processor.process_single_entry()
    breakpoint()


if __name__ == '__main__':
    info = MPInfo(protein_pdb="/vast/sx801/geometries/PDBBind2020_OG/RenumPDBs/10gs.renum.pdb",
    ligand_mol2="/vast/sx801/geometries/PDBBind2020_OG/PolarH/ligands/protonated/10gs_ligand.mol2")
    proc = PLPaddingStyleProcessor(info)
    proc.process_single_entry()
