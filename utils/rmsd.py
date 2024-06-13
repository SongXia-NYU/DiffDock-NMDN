from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile, Mol
import numpy as np
from spyrmsd import rmsd, molecule

from utils.utils_functions import TimeoutException, time_limit


def symmetry_rmsd_from_mols(mol_ref: Mol, mol_pred: Mol, timeout: int = 9999, **kwargs):
    if mol_ref is None or mol_pred is None:
        return np.inf
    
    ref_lig_pos = np.array(mol_ref.GetConformer().GetPositions())
    pred_lig_pos = mol_pred.GetConformer().GetPositions()

    try:
        with time_limit(timeout):
            mol_ref = molecule.Molecule.from_rdkit(mol_ref)
            # Hydrogens are masked out
            mol_ref_h_masker = (mol_ref.atomicnums != 1)
            mol_pred = molecule.Molecule.from_rdkit(mol_pred)
            mol_pred_z = mol_pred.atomicnums
            mol_pred_h_masker = (mol_pred_z != 1)
            mol_pred_adj_mtrtrix = mol_pred.adjacency_matrix
            try:
                computed_rmsd = rmsd.symmrmsd(
                    ref_lig_pos[mol_ref_h_masker, :],
                    pred_lig_pos[mol_pred_h_masker, :],
                    mol_ref.atomicnums[mol_ref_h_masker],
                    mol_pred_z[mol_pred_h_masker],
                    mol_ref.adjacency_matrix[mol_ref_h_masker, :][:, mol_ref_h_masker],
                    mol_pred_adj_mtrtrix[mol_pred_h_masker, :][:, mol_pred_h_masker],
                    **kwargs
                )
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                computed_rmsd = np.sqrt(((pred_lig_pos[mol_pred_h_masker, :] - ref_lig_pos[mol_ref_h_masker, :]) ** 2).sum(axis=-1).mean(axis=-1)).item()
    except TimeoutException as e:
        print("Timed out! Using non corrected RMSD..")
        computed_rmsd = np.sqrt(((pred_lig_pos[mol_pred_h_masker, :] - ref_lig_pos[mol_ref_h_masker, :]) ** 2).sum(axis=-1).mean(axis=-1)).item()
    return computed_rmsd