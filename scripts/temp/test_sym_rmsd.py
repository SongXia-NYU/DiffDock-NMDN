import rdkit
from rdkit.Chem import MolFromMol2File
from rdkit.Chem import rdMolTransforms
import numpy as np

from utils.rmsd import symmetry_rmsd_from_mols

def rot_ar_x(radi):
    return  np.array([[1, 0, 0, 0],
                      [0, np.cos(radi), -np.sin(radi), 0],
                      [0, np.sin(radi), np.cos(radi), 0],
                     [0, 0, 0, 1]], dtype=np.double)
 
def rot_ar_y(radi):
    return  np.array([[np.cos(radi), 0, np.sin(radi), 0],
                      [0, 1, 0, 0],
                      [-np.sin(radi), 0, np.cos(radi), 0],
                     [0, 0, 0, 1]], dtype=np.double)
 
def rot_ar_z(radi):
    return  np.array([[np.cos(radi), -np.sin(radi), 0, 0],
                      [np.sin(radi), np.cos(radi), 0, 0],
                      [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.double)
tforms = {0: rot_ar_x, 1: rot_ar_y, 2: rot_ar_z}

mol_ref = MolFromMol2File("/scratch/sx801/temp/compare_rmsd/1a30_ligand.mol2")
mol_1 = MolFromMol2File("/scratch/sx801/temp/compare_rmsd/1a30_decoys.mol2")

rmsd = symmetry_rmsd_from_mols(mol_ref, mol_1, minimize=True)
print(rmsd)

rdMolTransforms.TransformConformer(mol_1.GetConformer(0), tforms[0](2*np.pi/10))
rmsd = symmetry_rmsd_from_mols(mol_ref, mol_1, minimize=True)
print(rmsd)
