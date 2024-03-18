from collections import defaultdict
import os.path as osp
import os
from glob import glob
from typing import List, Union
import pandas as pd

class MerckFEPReader:
    def __init__(self, ds_root: str) -> None:
        self.ds_root = ds_root

        self.TARGETS = "cdk8  cmet  eg5  eg5_alternativeloop  hif2a  pfkfb3  shp2  syk  tnks2".split()
        for target in self.TARGETS:
            assert osp.isdir(osp.join(self.ds_root, target))

        self.info_df_query = {}

    def get_prot_src(self, target: str) -> str:
        pdb_files = glob(osp.join(self.ds_root, target, "*prepared.pdb"))
        assert len(pdb_files) == 1, pdb_files
        return pdb_files[0]

    def get_ligs_src(self, target: str) -> str:
        return osp.join(self.ds_root, target, "ligands.sdf")
    
    def get_prot_polarh(self, target: str) -> str:
        return osp.join(self.ds_root, target, "processed", "prot.polarh.pdb")
    
    def get_ligs_polarh(self, target: str) -> str:
        return osp.join(self.ds_root, target, "processed", "ligs.polarh.sdf")
    
    def get_ligs_diffdock_root(self) -> List[str]:
        return glob(osp.join(self.ds_root, "pose_diffdock", "raw_predicts", "*"))
    
    def get_ligs_diffdock_polarh(self, target: str, ligand_id: str) -> str:
        return osp.join(self.ds_root, "pose_diffdock", "polarh", f"{target}.{ligand_id}.polarh.sdf")
    
    def get_ligs_diffdock_polarhs(self) -> List[str]:
        return glob(osp.join(self.ds_root, "pose_diffdock", "polarh", f"*.polarh.sdf"))
    
    def query_exp_delta_g(self, target: str, ligand_id: Union[str, int]):
        def _dtype(og):
            try:
                og = int(float(og))
            except ValueError:
                pass
            return str(og)
        
        if target in self.info_df_query:
            info_df: pd.DataFrame = self.info_df_query[target]
        else:
            info_df = pd.read_csv(osp.join(self.ds_root, target, "results_5ns.csv"))
            info_df = info_df.astype({"Ligand": str})
            info_df["ligand_id"] = info_df["Ligand"].map(_dtype)
            info_df = info_df.set_index("ligand_id")
            self.info_df_query[target] = info_df

        return info_df.loc[ligand_id, "Exp. ΔG"]

if __name__ == "__main__":
    reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")
