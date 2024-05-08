from typing import List
import shutil
import pandas as pd
import os
import os.path as osp
from glob import glob
from tqdm import tqdm

store = "/vast/sx801/Diffdock_NMDN"

def prepare_casf_scoring():
    save_root = osp.join(store, "CASF-2016-scoring-ranking")
    os.makedirs(save_root, exist_ok=True)
    record_csv = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_549_run_2024-04-10_144554__079647/exp_pl_549_test_on_casf2016-blind-docking_2024-04-12_053305/record_name_447c71a3e82c3e366ae05cec76295735197c1c53d7c18101dc53ea1004dcf9d8.csv"
    record_df = pd.read_csv(record_csv)
    file_handles: List[str] = record_df["file_handle"].values.tolist()

    for fl in tqdm(file_handles):
        pdb, lig_rank = fl.split(".")
        lig_rank = int(lig_rank.split("lig_srcrank")[-1])
        src = f"/docking/pose_diffdock/raw_predicts/{pdb}/rank{lig_rank}_confidence*.sdf"
        src = glob(src)[0]
        dst = osp.join(save_root, f"{pdb}_diffdock_nmdn.sdf")
        shutil.copyfile(src, dst)

def prepare_casf_screening():
    save_root = osp.join(store, "CASF-2016-screening")
    os.makedirs(save_root, exist_ok=True)
    record_csvs = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_549_run_2024-04-10_144554__079647/exp_pl_549_test_on_casf2016-blind-screening_2024-04-12_053337/record_name_*.csv"
    record_csvs = glob(record_csvs)
    record_df = [pd.read_csv(record_csv) for record_csv in record_csvs]
    record_df = pd.concat(record_df, axis=0)
    file_handles: List[str] = record_df["file_handle"].values.tolist()

    for fl in tqdm(file_handles):
        pdbs, lig_rank = fl.split(".")
        lig_rank = int(lig_rank.split("lig_srcrank")[-1])
        tgt_pdb, lig_pdb = pdbs.split("_")
        src = f"/screening/pose_diffdock/raw_predicts//{pdbs}/rank{lig_rank}_confidence*.sdf"
        src = glob(src)[0]
        dst = osp.join(save_root, f"{tgt_pdb}/{pdbs}_diffdock_nmdn.sdf")
        os.makedirs(osp.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

def prepare_lit_pcba():
    for tgt in ["ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", 
"MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]:
        save_root = osp.join(store, "LIT-PCBA", tgt)
        os.makedirs(save_root, exist_ok=True)
        res_folder = f"/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_549_run_2024-04-10_144554__079647/exp_pl_549_test_on_{tgt.lower()}-diffdock_*"
        record_csvs = osp.join(res_folder, "record_name_*.csv")
        record_csvs = glob(record_csvs)
        record_df = [pd.read_csv(record_csv) for record_csv in record_csvs]
        record_df = pd.concat(record_df, axis=0)

        for i in tqdm(range(record_df.shape[0]), desc=tgt):
            this_info = record_df.iloc[i]
            fl = this_info["file_handle"]
            rank = this_info["rank"]
            src = f"/{tgt}/pose_diffdock/raw_predicts/{fl}/rank{rank}_confidence*.sdf"
            src = glob(src)[0]
            dst = osp.join(save_root, f"{fl}_diffdock_nmdn.sdf")
            shutil.copyfile(src, dst)


def prepare_merck_fep():
    save_root = osp.join(store, "MerckFEP")
    os.makedirs(save_root, exist_ok=True)
    record_csv = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_549_run_2024-04-10_144554__079647/exp_pl_549_test_on_merck_fep-diffdock_2024-04-23_190245/record_name_0d52e3f178485ca92ef206c64c182a1877bfcd24d62e0a620cf72cf236dd116b.csv"
    record_df = pd.read_csv(record_csv)
    file_handles: List[str] = record_df["file_handle"].values.tolist()

    for fl in tqdm(file_handles):
        rank = fl.split(".")[-1]
        fl = fl.split(f".{rank}")[0]
        if "Example" in fl:
            fl = fl.split("Example")[0] + "Example " + fl.split("Example")[1]
        srcdir = osp.join("/vast/sx801/geometries/fep-benchmark/pose_diffdock/raw_predicts", fl)
        src = osp.join(srcdir, f"{rank}_confidence*.sdf")
        src = glob(src)[0]
        dst = osp.join(save_root, f"{fl}_diffdock_nmdn.sdf")
        shutil.copyfile(src, dst)

if __name__ == "__main__":
    prepare_merck_fep()
