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
import pickle
from chemprop_kano.features.featurization import MolGraph

from sklearn.model_selection import train_test_split
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.qm9_reader import QM9Reader
from geometry_processors.process.mp_pyg_runner import MultiProcessPygRunner, proc_ligand

reader = QM9Reader("/QM9_M")

mol_g_arg = Namespace(atom_messages=False)

ds: Dict[str, MolGraph] = {}

for molid in tqdm(reader.molids):
    lig_sdf = reader.molid2qm_sdf(molid)
    with SDMolSupplier(lig_sdf, removeHs=False, sanitize=False) as supp:
        mol = supp[0]
    mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
    ds[molid] = mol_g


with open("/scratch/sx801/data/im_datasets/processed/qm9_qm_kano.pickle", "wb") as f:
    pickle.dump(ds, f)
print("finished")
