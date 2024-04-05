from glob import glob
import os.path as osp

target_proteins = glob("/vast/sx801/geometries/LIT-PCBA-DiffDock/*/????_protein.mol2")
target_pdbs = [osp.basename(p).split("_")[0] for p in target_proteins]
for pdb in target_pdbs:
    print(f"{pdb}", end=" ")
