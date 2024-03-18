import argparse
import json
import alphaspace2 as al
import mdtraj
from tqdm.contrib.concurrent import process_map
from glob import glob
import numpy as np
import os
import os.path as osp
import subprocess
import shutil

from prody import parsePDB, writePDB, AtomGroup


class AlphaSpace2Calculator:
    def __init__(self, src_pdb, dst_folder) -> None:
        self.src_pdb = src_pdb
        self.dst_folder = dst_folder

    def run(self):
        dst = self.dst_folder
        if osp.exists(dst):
            beta_atoms = osp.join(dst, "beta_atoms.pdb")
            alpha_atoms = osp.join(dst, "alpha_atoms.pdb")
            if osp.exists(beta_atoms) and osp.exists(alpha_atoms):
                print(f"{dst} finishes, skipping...")
                return
            shutil.rmtree(dst)

        self._run_alphaspace2()
        self._combine_atoms()
        self._cleanup()

    def _run_alphaspace2(self):
        dst = self.dst_folder
        src = self.src_pdb

        os.makedirs(dst)
        ss_prop = al.Snapshot()
        prot = mdtraj.load(src)
        ss_prop.run(prot)
        ss_prop.save(dst)

    def _combine_atoms(self):
        beta_atoms = glob(osp.join(self.dst_folder, "pockets", "*_beta.pdb"))
        beta_atoms = [parsePDB(beta_pdb) for beta_pdb in beta_atoms]
        beta_atoms = [ag for ag in beta_atoms if isinstance(ag, AtomGroup)]
        combined_pdb = sum(beta_atoms[1:], beta_atoms[0])
        combined_pdb.setTitle("AlphaSpace2 Beta Atoms")
        writePDB(osp.join(self.dst_folder, "beta_atoms.pdb"), combined_pdb)

        alpha_atoms = glob(osp.join(self.dst_folder, "pockets", "*_alpha.pdb"))
        alpha_atoms = [parsePDB(alpha_pdb) for alpha_pdb in alpha_atoms]
        alpha_atoms = [ag for ag in alpha_atoms if isinstance(ag, AtomGroup)]
        combined_pdb = sum(alpha_atoms[1:], alpha_atoms[0])
        combined_pdb.setTitle("AlphaSpace2 alpha Atoms")
        writePDB(osp.join(self.dst_folder, "alpha_atoms.pdb"), combined_pdb)

    def _cleanup(self):
        dst = self.dst_folder
        # remove and reorganize files
        subprocess.run("rm AS_Chimera*.py", cwd=dst, check=True, shell=True)
        subprocess.run("rm colors_*.txt", cwd=dst, check=True, shell=True)
        subprocess.run("rm -r pockets", cwd=dst, check=True, shell=True)

