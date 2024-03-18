from typing import List
import numpy as np
import torch
from argparse import Namespace
from typing import Dict
import rdkit
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.rdmolfiles import MolFromMol2File
from tqdm import tqdm
import os.path as osp
import yaml
from glob import glob
import pickle
from chemprop_kano.features.featurization import MolGraph

from sklearn.model_selection import train_test_split
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.qm9_reader import QM9Reader
from geometry_processors.process.mp_pyg_runner import MultiProcessPygRunner, proc_ligand

mol_g_arg = Namespace(atom_messages=False)

ds: Dict[str, MolGraph] = {}

for lig_sdf in tqdm(glob("/scratch/sx801/scripts/drug_descovery_perspective/data/mmff_geometries/conf_lowest/*.sdf")):
    molid = osp.basename(lig_sdf).split(".sdf")[0]
    with SDMolSupplier(lig_sdf, removeHs=False, sanitize=False) as supp:
        mol = supp[0]
    mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
    ds[molid] = mol_g


with open("/scratch/sx801/data/im_datasets/processed/esol_mmff_kano.pickle", "wb") as f:
    pickle.dump(ds, f)
print("finished")
