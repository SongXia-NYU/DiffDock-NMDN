from collections import defaultdict
import os.path as osp
import os
from glob import glob
from typing import List, Union
import pandas as pd

from geometry_processors.lazy_property import lazy_property

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

    # --------- DiffDock and NMDN rank1 --------- #
    def get_diffdock_rank1(self, target: str, ligand_id: str):
        return osp.join(self.ds_root, "pose_diffdock", "raw_predicts", f"{target}.{ligand_id}", "rank1.sdf")
    
    def get_nmdn_rank1(self, target: str, ligand_id: str):
        ligand_query = ligand_id
        if "Example" in ligand_query: 
            ligand_query = "".join(ligand_query.split(" "))
        ligand_query = f"{target}.{ligand_query}"
        if ligand_query not in self.nmdn_rank1_df.index:
            return self.get_diffdock_rank1(target, ligand_id)
        rank = self.nmdn_rank1_df.loc[ligand_query, "rank"]
        files = glob(osp.join(self.ds_root, "pose_diffdock", "raw_predicts", f"{target}.{ligand_id}", f"{rank}_confidence*.sdf"))
        assert len(files) == 1, files
        return files[0]

    @lazy_property
    def nmdn_rank1_df(self):
        nmdn_rank1_csv = osp.join(self.ds_root, "pose_diffdock", "nmdn_rank1.csv")
        if osp.exists(nmdn_rank1_csv):
            return pd.read_csv(nmdn_rank1_csv).set_index("file_handle")
        
        from utils.eval.TestedFolderReader import TestedFolderReader
        scoring_reader = TestedFolderReader("exp_pl_534_run_2024-01-22_211045__480688",
                    "exp_pl_534_test_on_merck_fep-diffdock_2024-02-29_182750",
                    "/scratch/sx801/scripts/DiffDock-NMDN")
        scoring_result = scoring_reader.result_mapper["test"]
        res_info = {"sample_id": scoring_result["sample_id"],
                    "pKd_score": scoring_result["PROP_PRED"].view(-1).numpy(),
                    "NMDN_score": scoring_result["MDN_LOGSUM_DIST2_REFDIST2"].view(-1).numpy()}
        res_df: pd.DataFrame = pd.DataFrame(res_info).set_index("sample_id")
        record: pd.DataFrame = scoring_reader.only_record().set_index("sample_id")
        res_df = res_df.join(record)
        
        res_df["rank"] = res_df["file_handle"].map(lambda s: s.split(".")[-1])
        res_df["file_handle"] = res_df["file_handle"].map(lambda s: ".".join(s.split(".")[:-1]))
        res_df = res_df.sort_values("NMDN_score", ascending=False).drop_duplicates("file_handle")
        res_df.to_csv(nmdn_rank1_csv, index=False)
        return res_df.set_index("file_handle")

if __name__ == "__main__":
    reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")
