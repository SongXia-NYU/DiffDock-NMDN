from typing import List
from utils.data.DummyIMDataset import DummyIMDataset
from torch_geometric.loader import DataLoader
import pickle
from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader, mol2_to_info_list
from geometry_processors.pl_dataset.ConfReader import Mol2Reader, ConfReader
from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory
import numpy as np
import os.path as osp
import pandas as pd

def debug_kano_pdbind_ds():
    pyg_ds = DummyIMDataset(data_root="/scratch/sx801/data/im_datasets/", dataset_name="PBind2020OG.polar.polar.implicit.min_dist.linf9.pyg", cfg=None)
    with open("/scratch/sx801/data/im_datasets/processed/pdbbind2020_og_kano.lite.pickle", "rb") as f:
        kano_ds: dict = pickle.load(f)
    for pygd in pyg_ds:
        if pygd.pdb not in kano_ds:
            continue
        kanod = kano_ds[pygd.pdb]
        assert kanod.n_atoms == pygd.R.shape[0], (kanod.n_atoms, pygd.R.shape)
    print("Great success!!")


class Same2DChecker:
    def __init__(self, mol2s: List[str]) -> None:
        self.mol2s = mol2s

        self.n_atoms: int = None
        self.elements: List[int] = None

    def check(self):
        for mol2 in self.mol2s:
            for i, info in enumerate(mol2_to_info_list(None, mol2, None)):
                reader, __ = ConfReaderFactory(info).get_lig_reader()
                success, err = self._check(reader)
                if success:
                    continue
                print(f"Error in {osp.basename(mol2)} idx {i}: {err}")
    
    def _check(self, reader: ConfReader):
        if self.n_atoms is None:
            assert self.elements is None, self.elements
            self.n_atoms = reader.n_atoms
            self.elements = reader.elements
            return True, ""
        
        success: bool = True
        err_msg = ""
        if self.n_atoms != reader.n_atoms:
            success = False
            err_msg += "n_atoms"
        if self.elements != reader.elements:
            success = False
            err_msg += ";elements"
        return success, err_msg

def debug_casf_docking():
    casf_reader = CASF2016Reader("/CASF-2016-cyang")
    for pdb in casf_reader.pdbs:
        print(f"Checking {pdb}")
        checker = Same2DChecker(casf_reader.pdb2dock_polarh_ligs(pdb))
        checker.check()
    # sad. The 2D graphs are not the same.

if __name__ == "__main__":
    debug_casf_docking()
