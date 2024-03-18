from functools import partial
import os
import subprocess
from typing import List, Union
import os.path as osp
import glob
import torch
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torch_scatter import scatter_add
from torch_geometric.data import Data

from geometry_processors.misc import solv_num_workers
from geometry_processors.pl_dataset.martini_mapper import MartiniMapper

"""
Scripts to calculate Solvent Accessible Surface Area for each atom by calling the MSMS software which is available at https://ccsb.scripps.edu/msms/.
WARNING: In every rare cases, the atom SASA is negative (e.g. some atoms in ["1z3v", "6d6t", "5h8b"]). 
    This is weird. Probably something is going on in the MSMS software. 
    I chose to ignore them by calling replacing the negative values with zero since they are very rare.
"""

MSMS_PATH = "/scratch/sx801/scripts/msms_i86_64Linux2_2.6.1"
PDB2XYZR = f"{MSMS_PATH}/pdb_to_xyzr"
MSMS = f"{MSMS_PATH}/msms.x86_64Linux2.2.6.1"

def _single_runner(pdb_file, save_root):
    calculator = SASASingleCalculator(pdb_file, save_root)
    try:
        return calculator.run()
    except Exception as e:
        print(e)
        calculator.remove_dir()

class SASABatchCalculator:
    def __init__(self, pdb_files, save_root) -> None:
        self.save_root = save_root

        if isinstance(pdb_files, str):
            pdb_files = glob.glob(pdb_files)
        self.pdb_files = pdb_files

    def run(self):
        rum_runner = partial(_single_runner, save_root=self.save_root)
        __, __, num_workers = solv_num_workers()
        process_map(rum_runner, self.pdb_files, max_workers=num_workers+4, chunksize=10)

    def run_sp(self):
        rum_runner = partial(_single_runner, save_root=self.save_root)
        for pdb_file in tqdm(self.pdb_files):
            rum_runner(pdb_file)

class SASASingleCalculator:
    """
    Calculate solvent accessible surface area (SASA) of a protein
    """
    def __init__(self, pdb_file, save_root) -> None:
        self.pdb_file = pdb_file
        self.save_root = save_root

        self.pdb_base = osp.basename(pdb_file).split(".pdb")[0]
        self.workdir = osp.join(save_root, self.pdb_base)
        self.xyzr_name = f"{self.pdb_base}.xyzr"
        self.out_name = f"{self.pdb_base}.area"
        self.log_name = f"{self.pdb_base}.log"

    def run(self):
        if osp.exists(osp.join(self.workdir, self.out_name)):
            print(f"{self.out_name} exists, exiting...")
            return
        os.makedirs(self.workdir, exist_ok=True)
        self.pdb2xyzr()
        self.msms()

    def pdb2xyzr(self):
        cmd = f"{PDB2XYZR} {self.pdb_file} > {self.xyzr_name}"
        subprocess.run(cmd, shell=True, check=True, cwd=self.workdir)

    def msms(self):
        cmd = f"{MSMS} -if {self.xyzr_name} -af {self.out_name} -probe_radius 1.0 -surface ases > {self.log_name} 2>&1 "
        subprocess.run(cmd, shell=True, check=True, cwd=self.workdir)

    def remove_dir(self):
        cmd = f"rm -r {self.workdir}"
        subprocess.run(cmd, shell=True, check=True)

class CGMartiniSASASummarizer:
    """
    Read sasa output and summarize it in torch_geometric.data format.
    The calculated SASA is atom-level, here we are going to sum them up into CGMartini-level.
    """
    def __init__(self, atom_pdb, cg_pdb, sasa_out) -> None:
        self.atom_pdb = atom_pdb
        self.cg_pdb = cg_pdb
        self.martini_mapper = MartiniMapper(atom_pdb, cg_pdb)
        self.sasa_out = sasa_out
        
        self.sasa_df = pd.read_csv(sasa_out, sep="\s+")

    def run(self):
        sasa_tensor = torch.as_tensor(self.sasa_df["sas_0"].values)
        # filling negative values with zero. see the "WARNING" comment at the top of this file.
        sasa_tensor[sasa_tensor<0] = 0.
        atom2martini_batch = self.martini_mapper.get_batch_index()
        assert sasa_tensor.shape == atom2martini_batch.shape

        ignore_mask = (atom2martini_batch == -1)
        # set those ignored entries to zero
        sasa_tensor[ignore_mask] = 0.
        # set the ignored index to max+1, then discard them
        atom2martini_batch[ignore_mask] = atom2martini_batch.max() + 1
        martini_sasa = scatter_add(sasa_tensor, atom2martini_batch, dim=0)
        martini_sasa = martini_sasa[:-1]
        
        assert martini_sasa.shape[0] == self.martini_mapper.cg_reader.numAtoms()
        
        pdb = osp.basename(self.atom_pdb).split(".")[0]
        sasa_data = Data(pdb=pdb, martini_sasa=martini_sasa)
        return sasa_data


if __name__ == "__main__":
    data_root = "/scratch/sx801/temp"
    summarizer = CGMartiniSASASummarizer("/scratch/sx801/temp/RenumPDBs/1a07.renum.pdb", "/scratch/sx801/temp/Martini/pdb/1a07.renum.martini.pdb",
        "/scratch/sx801/temp/SASA_features/1a07.renum/1a07.renum.area")
    summarizer.run()

