from collections import defaultdict
import os.path as osp
from glob import glob
from typing import List
import pandas as pd
from tqdm import tqdm

from geometry_processors.misc import lazy_property


class CASF2016BlindDocking:
    def __init__(self, droot: str) -> None:
        self.droot = droot
        self.dock_root = osp.join(droot, "docking_paper_models")
        self.screen_root = osp.join(droot, "screening")

    @lazy_property
    def dock_pdbs(self) -> List[str]:
        folders = glob(osp.join(self.dock_root, "pose_diffdock", "raw_predicts", "????"))
        pdbs = [osp.basename(folder) for folder in folders]
        return pdbs
    
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
    def dock_info(self) -> pd.DataFrame:
        dock_info_csv = osp.join(self.dock_root, "dock_info.csv")
        if osp.exists(dock_info_csv):
            return pd.read_csv(dock_info_csv).set_index("file_handle")
        
        dock_info_dict = defaultdict(lambda: [])
        for pdb_id in self.dock_pdbs:
            for raw_lig_f in glob(osp.join(self.dock_root, "pose_diffdock", "raw_predicts", pdb_id, "rank*_confidence*.sdf")):
                file_name = osp.basename(raw_lig_f)
                rank = int(file_name.split("_")[0][4:])
                confidence = float(file_name.split("_confidence")[-1].split(".sdf")[0])
                file_handle = f"{pdb_id}.lig_srcrank{rank}"
                dock_info_dict["file_handle"].append(file_handle)
                dock_info_dict["rank"].append(rank)
                dock_info_dict["confidence"].append(confidence)
                dock_info_dict["pdb_id"].append(pdb_id)
                dock_info_dict["raw_file_name"].append(file_name)
        dock_info_df = pd.DataFrame(dock_info_dict).set_index("file_handle")
        dock_info_df.to_csv(dock_info_csv)
        return dock_info_df
    
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
