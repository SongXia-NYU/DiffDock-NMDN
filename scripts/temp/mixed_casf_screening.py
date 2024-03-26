# NMDN to select pose and pkd score to score/rank/screen
from glob import glob
import os
import os.path as osp
import pandas as pd
import json

from utils.scores.casf_scores import calc_screening_score

workdir = "/scratch/sx801/temp/pl_534_mixed_mdn"
savedir = f"{workdir}/mixed_nmdn_pkd/pred_data"
os.makedirs(savedir, exist_ok=True)

def conv():
    nmdn_preds = glob(osp.join(workdir, "screening-MDN_LOGSUM_DIST2_REFDIST2/pred_data/????_score.dat"))
    pdbs = [osp.basename(pred).split("_")[0] for pred in nmdn_preds]

    for pdb in pdbs:
        nmdn_pred = osp.join(workdir, f"screening-MDN_LOGSUM_DIST2_REFDIST2/pred_data/{pdb}_score.dat")
        nmdn_pred_df = pd.read_csv(nmdn_pred, sep=" ").set_index("#code_ligand_num").rename({"score": "nmdn_score"}, axis=1)
        pkd_pred = osp.join(workdir, f"screening-pKd/pred_data/{pdb}_score.dat")
        pkd_pred_df = pd.read_csv(pkd_pred, sep=" ").set_index("#code_ligand_num")
        pred_df = nmdn_pred_df.join(pkd_pred_df).reset_index()
        pred_df["lig_pdb"] = pred_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
        pred_df = pred_df.sort_values("nmdn_score", ascending=False).drop_duplicates("lig_pdb")
        pred_df = pred_df[["#code_ligand_num", "score"]]
        pred_df.to_csv(f"{savedir}/{pdb}_score.dat", sep=" ", index=False)

def score():
    result_summary = calc_screening_score(f"{workdir}/mixed_nmdn_pkd", "pred_data", "screening")
    with open(f"{workdir}/mixed_nmdn_pkd/result.json", "w") as f:
        json.dump(result_summary, f)

if __name__ == "__main__":
    score()
