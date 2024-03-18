from argparse import Namespace
from typing import Dict
import rdkit
from rdkit.Chem.rdmolfiles import MolFromMol2File
from tqdm import tqdm
import os.path as osp
import yaml
import pickle

from chemprop_kano.features.featurization import MolGraph
from geometry_processors.pl_dataset.pdb2020_ds_reader import PDB2020DSReader

DS_ROOT = "/PDBBind2020_OG"
ds_reader = PDB2020DSReader(DS_ROOT)

mol_g_arg = Namespace(atom_messages=False)

ds: Dict[str, MolGraph] = {}

for pdb in tqdm(ds_reader.iter_pdbs()):
    lig_mol2: str = ds_reader.pdb2proton_polar_lig(pdb)
    if not osp.exists(lig_mol2):
        continue
    mol = MolFromMol2File(lig_mol2, removeHs=False, sanitize=False)
    mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
    ds[pdb] = mol_g


with open("/scratch/sx801/data/im_datasets/processed/pdbbind2020_og_kano.pickle", "wb") as f:
    pickle.dump(ds, f)
print("finished")

