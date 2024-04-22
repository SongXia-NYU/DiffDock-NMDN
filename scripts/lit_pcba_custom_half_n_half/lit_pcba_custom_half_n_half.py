# pKd select top 0.5% and NMDN select top 0.5%
from collections import defaultdict
import json
import os
import os.path as osp
import subprocess
import pandas as pd
import math

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen
from geometry_processors.pl_dataset.lit_pcba_reader import TARGETS

nmdn_src_root = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/lit-pcba-scores"
pkd_src_root = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_549_run_2024-04-10_144554__079647/lit-pcba-scores"
sname = "pl_534_549"

for target in TARGETS:
    workdir = osp.join("/scratch/sx801/scripts/DiffDock-NMDN/scripts/lit_pcba_custom_half_n_half", sname, target)
    os.makedirs(workdir, exist_ok=True)

    nmdn_best_score_tar = osp.join(nmdn_src_root, f"{target.lower()}-diffdock", "best_score_csv.tar.gz")
    subprocess.run(f"tar xvf {nmdn_best_score_tar} ", shell=True, check=True, cwd=workdir)

    pkd_best_score_tar = osp.join(pkd_src_root, f"{target.lower()}-diffdock", "best_score_csv.tar.gz")
    subprocess.run(f"tar xvf {pkd_best_score_tar} ", shell=True, check=True, cwd=workdir)

    nmdn_best_df = pd.read_csv(osp.join(workdir, "best_score_score_MDN_LOGSUM_DIST2_REFDIST2.csv"))
    pkd_best_df = pd.read_csv(osp.join(workdir, "best_score_score.csv"))

    nmdn_best_df = nmdn_best_df.sort_values(by="best_score", ascending=False)
    pkd_best_df = pkd_best_df.sort_values(by="best_score", ascending=False)

    n_interested = math.ceil(nmdn_best_df.shape[0] * 0.01)
    for i in range(n_interested//2, nmdn_best_df.shape[0]):
        sel = set(nmdn_best_df["og_id"][:i].values.reshape(-1).tolist()).union(pkd_best_df["og_id"][:i].values.reshape(-1).tolist())
        if len(sel) >= n_interested:
            break
    all_og_ids = set(nmdn_best_df["og_id"].values.reshape(-1).tolist())
    un_sel = all_og_ids.difference(sel)

    out_df = {"og_id": list(sel), "best_score": [1.] * len(sel)}
    out_df["og_id"].extend(list(un_sel))
    out_df["best_score"].extend([0. for __ in range(len(un_sel))])

    out_df = pd.DataFrame(out_df)
    out_df.to_csv(osp.join(workdir, "half_n_half.csv"), index=False)

    DROOT = f"/LIT-PCBA/{target}"
    half_half_calc = LIT_PCBA_Screen(None, DROOT, workdir)
    half_half_calc.run(out_df)

    print("Finished", target)

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", 
                   "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]
out_info: dict = defaultdict(lambda: [])
for target in TARGETS:
    workdir = osp.join("/scratch/sx801/scripts/DiffDock-NMDN/scripts/lit_pcba_custom_half_n_half", sname, target)
    out_info["target"].append(target)
    with open(osp.join(workdir, "screening_score.json")) as f:
        out_info["EF1"].append(json.load(f)["EF1"])
out_df: pd.DataFrame = pd.DataFrame(out_info)
out_df.to_csv(f"{sname}.csv")
out_df.to_excel(f"{sname}.xlsx")