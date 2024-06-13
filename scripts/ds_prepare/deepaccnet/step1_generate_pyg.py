import argparse
from glob import glob
import os
import os.path as osp

from typing import List

from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.process.mp_pyg_runner import MultiProcessPygRunner, proc_single_protein_implicit_mindist

# -----------------You need to modify the following variables----------------- #
# this is where you store all your pdb files
PDB_ROOT = "/scratch/sx801/temp/DeepAccNet_sample"
# this is the name of the data set
DS_NAME = "DeepAccNet_test"
# this is a temporary folder for multi-processing, this folder is safe to delete after running
CHUNK_ROOT = "/scratch/sx801/temp/chunks"
# this is where you save the running results
SAVE_ROOT = "/scratch/sx801/temp/data"
# -----------------------------------------------------------------------------#

SAVE_ROOT = osp.join(SAVE_ROOT, "processed")
os.makedirs(CHUNK_ROOT, exist_ok=True)
os.makedirs(SAVE_ROOT, exist_ok=True)
if PDB_ROOT == "/scratch/sx801/temp/DeepAccNet_sample":
    print("WARNING: you are using an example folder containing only 88 pdb files")

pdb_files = glob(osp.join(PDB_ROOT, "*_protein.pdb"))

info_list: List[MPInfo] = []
for pdb_file in pdb_files:
    this_info = MPInfo(protein_pdb=pdb_file)
    info_list.append(this_info)

runner = MultiProcessPygRunner(info_list=info_list, proc_fn=proc_single_protein_implicit_mindist,
                               ds_name=DS_NAME, chunk_root=CHUNK_ROOT, save_root=SAVE_ROOT)
runner.run_chunks()
runner.run_collate()
