from typing import List
import torch
from geometry_processors.process.sasa_calculator import SASA_PL_Calculator, SASASingleCalculator
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from utils.DataPrepareUtils import n_rotatable_bonds
from utils.data.MolFileDataset import SDFDataset
from utils.eval.predict import EnsPredictor
import argparse
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id = args.array_id

info_df = pd.read_csv("/scratch/sx801/cache/rmsd_csv/JOB-pdbbind_yaowen_nonbinders.rmsd.csv")
nonbinder_df = info_df[info_df["file_handle"].map(lambda s: s.startswith("nonbinders."))]
pdbbind_df = info_df[info_df["file_handle"].map(lambda s: not s.startswith("nonbinders."))]

nonbinder_df = np.array_split(nonbinder_df, 20)[array_id]
pdbbind_df = np.array_split(pdbbind_df, 20)[array_id]

atomprop_predictor = EnsPredictor("/scratch/sx801/scripts/physnet-dimenet1/MartiniDock/pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*")

def atom_sol_from_lig(lig_file):
    try:
        sdf_ds = SDFDataset([lig_file])
        dl = DataLoader(sdf_ds, batch_size=1)
        allh_batch = next(iter(dl))
        model_pred = atomprop_predictor.predict_nograd(allh_batch)
        phys_atom_prop = model_pred["atom_prop"]

        mol = sdf_ds.get_data_reader(sdf_ds.mol_files[0]).mol
        is_needed: List[bool] = []
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                is_needed.append(True)
                continue
            # hydrogen atom only has one neighbor
            neighbor = list(atom.GetNeighbors())[0]
            if neighbor.GetAtomicNum() != 6:
                is_needed.append(True)
                continue
            atom_idx = atom.GetIdx()
            neighbor_idx = neighbor.GetIdx()
            phys_atom_prop[neighbor_idx, :] += phys_atom_prop[atom_idx, :]
            is_needed.append(False)
        return phys_atom_prop[is_needed, :]
    except Exception as e:
        print(e)
        return None

for i in tqdm(range(pdbbind_df.shape[0])):
    this_info = pdbbind_df.iloc[i]
    fl = this_info["file_handle"]
    if osp.exists(f"/vast/sx801/temp_atomsol/{fl}.pth"): continue

    pdb = fl.split(".")[0]
    fname = ".".join(fl.split(".")[1:]) + ".sdf"
    lig_file = f"/PDBBind2020_DiffDock-sampled/raw_predicts/{pdb}/{fname}"
    torch.save(atom_sol_from_lig(lig_file), f"/vast/sx801/temp_atomsol/{fl}.pth")
    

nonbinder_info_df = pd.read_csv("/vast/sx801/geometries/Yaowen_nonbinders/sampled_pl_info.csv").set_index("file_handle")
for i in tqdm(range(nonbinder_df.shape[0])):
    this_info = nonbinder_df.iloc[i]
    lig_file = this_info["ligand_file"]
    lig_file = lig_file.replace("/vast/sx801/geometries/", "/")
    fl = this_info["file_handle"]
    if osp.exists(f"/vast/sx801/temp_atomsol/{fl}.pth"): continue

    fl_query = ".".join(fl.split(".")[:-1])
    pdb = nonbinder_info_df.loc[fl_query, "pdb_id"]
    torch.save(atom_sol_from_lig(lig_file), f"/vast/sx801/temp_atomsol/{fl}.pth")
