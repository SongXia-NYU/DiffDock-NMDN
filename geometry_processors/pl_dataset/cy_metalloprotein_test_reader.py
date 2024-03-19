from glob import glob
import os.path as osp
import pandas as pd
import yaml
from typing import Dict

from geometry_processors.lazy_property import lazy_property
from geometry_processors.pl_dataset.prot_utils import pdb2seq


class MetalloProteinReader:
    def __init__(self, droot: str) -> None:
        self.droot = droot

    def pdb2prot_pdbqt(self, pdb: str):
        return osp.join(self.droot, "wat", "protein_pdbqt", f"{pdb}_protein.pdbqt")
    
    def pdb2prot_polarh(self, pdb: str):
        return osp.join(self.droot, "wat", "protein_polarh", f"{pdb}_protein.polarh.pdb")
    
    def pdb2lig_pdbqt(self, pdb: str):
        return osp.join(self.droot, "dry", "ligand_pdbqt", f"{pdb}_ligand.pdbqt")
    
    def pdb2lig_polarh_sdf(self, pdb: str):
        return osp.join(self.droot, "dry", "ligand_polarh", f"{pdb}_ligand.polar.sdf")

    @lazy_property
    def pdbs(self):
        protein_pdbqts = glob(osp.join(self.droot, "wat", "protein_pdbqt", "????_protein.pdbqt"))
        pdbs = [osp.basename(s).split("_")[0] for s in protein_pdbqts]
        return pdbs
    
    @lazy_property
    def info_df(self) -> pd.DataFrame:
        info_csv: str = osp.join(self.droot, "ci1c00737_si_003.csv")
        return pd.read_csv(info_csv).set_index("pdb")
    
    @lazy_property
    def seq_info(self) -> Dict[str, str]:
        seq_info_yaml = osp.join(self.droot, "seq_info.yaml")
        if osp.exists(seq_info_yaml):
            with open(seq_info_yaml) as f:
                return yaml.safe_load(f)
            
        seq_info: Dict[str, str] = {}
        for pdb in self.pdbs:
            prot_pdb = self.pdb2prot_polarh(pdb)
            seq_info[pdb] = pdb2seq(prot_pdb)
        with open(seq_info_yaml, "w") as f:
            yaml.safe_dump(seq_info, f)
        return seq_info
