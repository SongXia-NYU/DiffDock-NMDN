from collections import defaultdict
import os.path as osp
from glob import glob
from typing import List
import pandas as pd
from tqdm import tqdm

from geometry_processors.lazy_property import lazy_property


class CASF2016BlindDocking:
    def __init__(self, droot: str) -> None:
        self.droot = droot
        self.dock_root = "/scratch/sx801/scripts/raw_data/diffdock/test/infer_casf_dock021_dock022_2023-06-25_002147"
        self.screen_root = osp.join(droot, "screening")

     # ------------ Docking results only ------------ #
    @lazy_property
    def dock_pdbs(self) -> List[str]:
        folders = glob(osp.join(self.dock_root, "raw_predicts", "????"))
        pdbs = [osp.basename(folder) for folder in folders]
        return pdbs
    
    def get_docking_diffdock_rank1(self, pdb: str):
        return osp.join(self.dock_root, "raw_predicts", pdb, "rank1.sdf")
    
    def get_docking_nmdn_rank1(self, pdb: str):
        if pdb not in self.docking_nmdn_score_df.index:
            return self.get_docking_diffdock_rank1(pdb)
        src_file: str = self.docking_nmdn_score_df.loc[pdb, "src"]
        src_file_name = osp.basename(src_file)
        return osp.join(self.dock_root, "raw_predicts", pdb, src_file_name)

    @lazy_property
    def docking_nmdn_score_df(self) -> pd.DataFrame:
        nmdn_csv = "/scratch/sx801/scripts/raw_data/diffdock/test/infer_exp_casf_dock021_exp_pl_534_2023-06-25_002147/pred.csv"
        nmdn_df = pd.read_csv(nmdn_csv)
        nmdn_df["pdb_id"] = nmdn_df["file_handle"].map(lambda s: s.split(".")[0])
        nmdn_df = nmdn_df.sort_values(by="MDN_LOGSUM_DIST2_REFDIST2", ascending=False).drop_duplicates("pdb_id").set_index("pdb_id")
        return nmdn_df
    
    # ------------ Screening results only ------------ #
    @lazy_property
    def screen_pdbs(self) -> List[str]:
        folders = glob(osp.join(self.screen_root, "pose_diffdock", "raw_predicts", "????_????"))
        pdbs = [osp.basename(folder) for folder in folders]
        return pdbs
    
    def info_entry2lig_polarh(self, info_entry: pd.Series):
        file_handle = info_entry.name
        pdb_id = info_entry["pdb_id"]
        droot = self.dock_root if len(pdb_id) == 4 else self.screen_root
        return osp.join(droot, "pose_diffdock", "polarh", pdb_id, f"{file_handle}.sdf")
    
    def info_entry2lig_raw(self, info_entry: pd.Series):
        pdb_id = info_entry["pdb_id"]
        file_name = info_entry["raw_file_name"]
        droot = self.dock_root if len(pdb_id) == 4 else self.screen_root
        return osp.join(droot, "pose_diffdock", "raw_predicts", pdb_id, f"{file_name}")
    
    @lazy_property
    def screen_info(self) -> pd.DataFrame:
        screen_info_csv = osp.join(self.screen_root, "screen_info.csv")
        if osp.exists(screen_info_csv):
            return pd.read_csv(screen_info_csv).set_index("file_handle")
        
        screen_info_dict = defaultdict(lambda: [])
        for pdb_id in tqdm(self.screen_pdbs):
            for raw_lig_f in glob(osp.join(self.screen_root, "pose_diffdock", "raw_predicts", pdb_id, "rank*_confidence*.sdf")):
                file_name = osp.basename(raw_lig_f)
                rank = int(file_name.split("_")[0][4:])
                confidence = float(file_name.split("_confidence")[-1].split(".sdf")[0])
                file_handle = f"{pdb_id}.lig_srcrank{rank}"
                screen_info_dict["file_handle"].append(file_handle)
                screen_info_dict["rank"].append(rank)
                screen_info_dict["confidence"].append(confidence)
                screen_info_dict["pdb_id"].append(pdb_id)
                screen_info_dict["raw_file_name"].append(file_name)
        screen_info_df = pd.DataFrame(screen_info_dict).set_index("file_handle")
        screen_info_df.to_csv(screen_info_csv)
        return screen_info_df
    
    def get_screening_diffdock_rank1(self, tgt_pdb: str, lig_pdb: str):
        return osp.join(self.screen_root, "pose_diffdock", "raw_predicts", f"{tgt_pdb}_{lig_pdb}", "rank1.sdf")
    
    def get_screening_nmdn_rank1(self, tgt_pdb: str, lig_pdb: str):
        lig_str = self.screening_nmdn_score_df.loc[tgt_pdb, lig_pdb]["#code_ligand_num"]
        lig_rank = int(lig_str.split("_")[-1])
        possible_files = glob(osp.join(self.screen_root, "pose_diffdock", "raw_predicts", f"{tgt_pdb}_{lig_pdb}", f"rank{lig_rank}_confidence*.sdf"))
        return possible_files[0]
    
    @lazy_property
    def screening_nmdn_score_df(self) -> pd.DataFrame:
        droot = "/vast/sx801/geometries/CASF-2016-BlindDocking/screening/exp_pl534_screening_scores/screening_scores_MDN_LOGSUM_DIST2_REFDIST2"
        concat_dfs = []
        for target_df in glob(osp.join(droot, "????_score.dat")):
            target_pdb = osp.basename(target_df).split("_")[0]
            score_df = pd.read_csv(target_df, sep=" ")
            score_df["lig_pdb"] = score_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
            score_df = score_df.sort_values("score", ascending=False).drop_duplicates("lig_pdb")
            score_df["target_pdb"] = [target_pdb] * score_df.shape[0]
            concat_dfs.append(score_df)
        concat_dfs = pd.concat(concat_dfs).set_index(["target_pdb", "lig_pdb"])
        return concat_dfs
