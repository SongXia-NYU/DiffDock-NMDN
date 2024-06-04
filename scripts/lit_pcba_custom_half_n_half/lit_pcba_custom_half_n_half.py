# pKd select top 0.5% and NMDN select top 0.5%
from collections import defaultdict
from glob import glob
import json
import os
import os.path as osp
import subprocess
import pandas as pd
import math

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen
from geometry_processors.pl_dataset.lit_pcba_reader import TARGETS

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

nmdn_src_root = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/lit-pcba-scores"
# pkd_src_root = "/scratch/sx801/scripts/raw_data/exp_pl_newcfg/exp_pl_558_run_2024-05-12_132158__944421/lit-pcba-scores"
pkd_src_root = "/scratch/sx801/scripts/DiffDock-NMDN/scripts/benchmark-vina/preds"
sname = "pl_534_vinardo"

def get_pkd_best_score(pkd_src_root, workdir, target):
    if pkd_src_root.endswith("/lit-pcba-scores"):
        pkd_best_score_tar = osp.join(pkd_src_root, f"{target.lower()}-diffdock", "best_score_csv.tar.gz")
        subprocess.run(f"tar xvf {pkd_best_score_tar} ", shell=True, check=True, cwd=workdir)
        pkd_best_df = pd.read_csv(osp.join(workdir, "best_score_score.csv"))
        pkd_best_df = pkd_best_df.sort_values(by="best_score", ascending=False)
        return pkd_best_df

    if osp.exists(osp.join(pkd_src_root, f"{target}.csv")):
        pkd_best_df = pd.read_csv(osp.join(pkd_src_root, f"{target}.csv"))[["fl", "vinardo_affinity"]]
    else:
        pred_csvs = osp.join(pkd_src_root, target, "*.csv")
        pred_csvs = glob(pred_csvs)
        pkd_best_df = pd.concat([pd.read_csv(csv) for csv in pred_csvs])[["fl", "vinardo_affinity"]]
    pkd_best_df["best_score"] = pkd_best_df["vinardo_affinity"]/pKd2deltaG
    pkd_best_df["og_id"] = pkd_best_df["fl"].map(lambda s: int(s.split("_")[0]))
    pkd_best_df = pkd_best_df.sort_values(by="best_score", ascending=False)[["og_id", "best_score"]]
    return pkd_best_df


for target in TARGETS:
    workdir = osp.join("/scratch/sx801/scripts/Protein-NMDN/scripts/lit_pcba_custom_half_n_half", sname, target)
    os.makedirs(workdir, exist_ok=True)

    nmdn_best_score_tar = osp.join(nmdn_src_root, f"{target.lower()}-diffdock", "best_score_csv.tar.gz")
    subprocess.run(f"tar xvf {nmdn_best_score_tar} ", shell=True, check=True, cwd=workdir)
    nmdn_best_df = pd.read_csv(osp.join(workdir, "best_score_score_MDN_LOGSUM_DIST2_REFDIST2.csv"))
    nmdn_best_df = nmdn_best_df.sort_values(by="best_score", ascending=False)

    pkd_best_df = get_pkd_best_score(pkd_src_root, workdir, target)

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
    workdir = osp.join("/scratch/sx801/scripts/Protein-NMDN/scripts/lit_pcba_custom_half_n_half", sname, target)
    out_info["target"].append(target)
    with open(osp.join(workdir, "screening_score.json")) as f:
        out_info["EF1"].append(json.load(f)["EF1"])
out_df: pd.DataFrame = pd.DataFrame(out_info)
out_df.to_csv(f"{sname}.csv")
out_df.to_excel(f"{sname}.xlsx")