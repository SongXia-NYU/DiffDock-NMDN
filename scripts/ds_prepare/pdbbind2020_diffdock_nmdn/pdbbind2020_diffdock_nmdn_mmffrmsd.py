import argparse
from collections import defaultdict
from copy import deepcopy
from glob import glob
import os.path as osp
import tempfile
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AddHs
import numpy as np
import pandas as pd

from geometry_processors.misc import ff_optimize
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.pdb2020_ds_reader import PDB2020DSReader
from geometry_processors.process.mp_pyg_runner import ArrayPygRunner, proc_hetero_graph
from geometry_processors.process.proc_hydrogen import LigPolarConverter
from utils.rmsd import symmetry_rmsd_from_mols

reader = PDB2020DSReader("/PDBBind2020_OG")
DS_NAME = "PBind2020OG.diffdock-nmdn.hetero.polar.polar"

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id: int = args.array_id

def gen_mmff_opt():
    pdb_folders = glob(f"/vast/sx801/geometries/PDBBind2020_DiffDock-sampled/raw_predicts/????")
    pdb_ids = [osp.basename(pdb_folder) for pdb_folder in pdb_folders]
    pdb_ids.sort()
    pdb_ids = np.array_split(pdb_ids, 20)[array_id]
    rmsd_info = defaultdict(lambda: [])

    for pdb_id in tqdm(pdb_ids):
        for raw_lig in glob(f"/vast/sx801/geometries/PDBBind2020_DiffDock-sampled/raw_predicts/{pdb_id}/rank*_confidence*.sdf"):
            fname = osp.basename(raw_lig)
            polarh_fname = f"{pdb_id}.{fname}"
            file_handle = polarh_fname.split(".sdf")[0]
            suppl = Chem.SDMolSupplier(raw_lig, removeHs=True)
            mol = suppl[0]
            if mol is None: continue
            mol = AddHs(mol, addCoords=True)
            dst_mol = deepcopy(mol)
            try:
                ff_optimize(dst_mol, [0])
            except Exception as e:
                continue
            rmsd = symmetry_rmsd_from_mols(mol, dst_mol, 1)
            rmsd_info["file_handle"].append(file_handle)
            rmsd_info["rmsd"].append(rmsd)

    rmsd_df = pd.DataFrame(rmsd_info)
    rmsd_df.to_csv(f"/vast/sx801/geometries/PDBBind2020_DiffDock-sampled/mmff.rmsd.info/chunk_{array_id}.csv", index=False)
                
if __name__ == "__main__":
    gen_mmff_opt()
