from dataclasses import dataclass
import pandas as pd
import os.path as osp
from typing import Optional, Union
import numpy as np
import torch
import rdkit
from rdkit.Chem.rdchem import Mol


def csv2input_list(csv_name, root):
    sub_name = osp.basename(csv_name).split(".csv")[0]
    if sub_name in ["binder5_dry", "binder5_wat"]:
        process_fn = _binder5
    elif sub_name in ["binder4_dry", "binder4_wat"]:
        process_fn = _binder4
    elif sub_name in ["CSAR_decoy_part1", "CSAR_decoy_part2"]:
        process_fn = _csar_decoy
    elif sub_name in ["binder1_dry", "binder1_wat", "binder2_dry", "binder2_wat", "binder3_dry", "binder3_wat",
                      "PDBbind_refined_dry", "PDBbind_refined_wat", "val_crystal", "val_local_dry", "val_local_wat",
                      "test_CASF_2016_crystal"]:
        process_fn = _binder1to3
    elif sub_name in ["CSAR_dry"]:
        process_fn = _csar_dry
    elif sub_name in ["E2E_dock_dry", "E2E_dock_wat", "val_E2E_dock_dry", "val_E2E_dock_wat"]:
        process_fn = _e2e_dock
    elif sub_name in ["CASF-2016", "train_set"]:
        # Those are from the newly curated PDB Bind 2020
        process_fn = _pdb_bind2020
    else:
        raise ValueError(f"Invalid csv: {csv_name}")

    split = osp.basename(osp.dirname(csv_name))

    df = pd.read_csv(csv_name)

    out = []
    for i in range(df.shape[0]):
        this_series = df.iloc[i]
        out.append(process_fn(this_series, sub_name, root, split))
    return out


def _pdb_bind2020(series, sub_name, root, split="train"):
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", sub_name, f"{sub_name}_protein", f"{pdb}_protein.pdb")
    ligand_sdf = osp.join(root, "structure", sub_name, f"{sub_name}_sdf", f"{pdb}_ligand.sdf")
    save_name = osp.join(root, "single_pygs", sub_name, f"{pdb}_pl.pyg")

    return MPInfo(protein_pdb, None, pKd, split, sub_name, save_name, ligand_sdf=ligand_sdf)


def _binder5(series, sub_name, root, split="train"):
    ligand_id = series["ligand_id"]
    ligand_id = str(ligand_id).zfill(5)
    o_index = series["o_index"]
    pdb = series["refPDB"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}_protein.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "pose", f"{ligand_id}_{pdb}_{o_index}.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


def _binder4(series, sub_name, root, split="train"):
    o_index = series["o_index"]
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}_protein.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "pose", f"{pdb}_docked_{o_index}.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


def _binder1to3(series, sub_name, root, split="train"):
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}_protein.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "pose", f"{pdb}_ligand.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


def _csar_decoy(series, sub_name, root, split="train"):
    o_index = series["o_index"]
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "pose", f"{pdb}_decoys_{o_index}.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


def _csar_dry(series, sub_name, root, split="train"):
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "pose", f"{pdb}_ligand.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


def _e2e_dock(series, sub_name, root, split="train"):
    pdb = series["pdb"]
    pKd = series["pKd"]

    protein_pdb = osp.join(root, "structure", "train", sub_name, "protein", f"{pdb}_protein.pdb")
    ligand_pdb = osp.join(root, "structure", "train", sub_name, "docked_pose", f"{pdb}.pdb")

    return MPInfo(protein_pdb, ligand_pdb, pKd, split, sub_name)


class MPInfo:
    """
    Input information for multiprocessing
    """
    def __init__(self, **kwargs):
        # -------------Protein geometry information------------- #
        self.protein_pdb: Optional[str] = kwargs.pop("protein_pdb", None)
        self.beta_atom_pdb: Optional[str] = kwargs.pop("beta_atom_pdb", None)
        # it is only used in preparing multi-resolution dataset with Martini Beads and AAs
        self.protein_atom_pdb: Optional[str] = kwargs.pop("protein_atom_pdb", None)
        self.protein_pdbqt: Optional[str] = kwargs.pop("protein_pdbqt", None)
        # used in protein-protein interaction prediction in Xuhang's project
        self.protein_chain1_pdb: Optional[str] = kwargs.pop("protein_chain1_pdb", None)
        self.protein_chain2_pdb: Optional[str] = kwargs.pop("protein_chain2_pdb", None)
        self.dockground2_combined_pdb: Optional[str] = kwargs.pop("dockground2_combined_pdb", None)
        self.chains_combined_pdb: Optional[str] = kwargs.pop("chains_combined_pdb", None)

        # -------------Ligand geometry information------------- #
        self.ligand_sdf: Optional[str] = kwargs.pop("ligand_sdf", None)
        self.ligand_mol2: Optional[str] = kwargs.pop("ligand_mol2", None)
        self.ligand_pdb: Optional[str] = kwargs.pop("ligand_pdb", None)
        self.ligand_pdbqt: Optional[str] = kwargs.pop("ligand_pdbqt", None)
        self.ligand_linf9_opt: Optional[str] = kwargs.pop("ligand_linf9_opt", None)
        # you can pass a rdkit mol object as well
        self.mol: Optional[Mol] = kwargs.pop("mol", None)
        self.overwrite_lig_pos: Optional[Union[np.ndarray, torch.Tensor]] = kwargs.pop("overwrite_lig_pos", None)

        # -------------Geometry modifying arguments------------- #
        self.discard_p_info: bool = kwargs.pop("discard_p_info", False)
        self.remove_h: bool = kwargs.pop("remove_h", False)

        # -------------Cutoff information------------- #
        self.cutoff_pl: float = kwargs.pop("cutoff_pl", 10.)
        self.cutoff_pp: float = kwargs.pop("cutoff_pp", 10.)
        self.cutoff_protein: float = kwargs.pop("cutoff_protein", 10.)
        self.cutoff_ligand: float = kwargs.pop("cutoff_ligand", 10.)

        # -------------Misc------------- #
        self.ligand_id: Optional[str] = kwargs.pop("ligand_id", None)
        self.pyg_name: Optional[str] = kwargs.pop("pyg_name", None)
        self.sub_name: Optional[str] = kwargs.pop("sub_name", None)
        self.split: Optional[str] = kwargs.pop("split", None)

        # labels that model needs to predict
        self.labels = kwargs

