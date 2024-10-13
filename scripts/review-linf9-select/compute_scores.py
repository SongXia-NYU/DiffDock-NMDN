from typing import Dict
import pandas as pd
import yaml
import os
import os.path as osp
from glob import glob
import argparse

from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader
from geometry_processors.post_analysis import plot_scatter_info
from utils.scores.casf_blind_scores import score_rank_power
from utils.scores.casf_scores import calc_screening_score
from utils.scores.merck_fep_scores import MerckFEPScoreCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--score_name", type=str)
args = parser.parse_args()
SCORE_NAME = args.score_name

def casf_score_rank():
    for score_name in [f"{SCORE_NAME}_score"]:
        out_df = pd.read_csv(f"./results_casf_docking/casf_{SCORE_NAME}.csv").rename({"pdb": "pdb_id"}, axis=1)
        out_df = out_df.sort_values(by=score_name, ascending=False).drop_duplicates("pdb_id")
        out_df["score"] = out_df[score_name]
        scores = score_rank_power(out_df, ".", f"casf_{SCORE_NAME}.png")
        with open(f"./docking_{SCORE_NAME}.yaml", "w") as f:
            yaml.safe_dump(scores, f)

def casf_screen():
    array_result_csvs = glob(f"./results_casf_screening/{SCORE_NAME}_raw_preds/casf_screen_{SCORE_NAME}.*.csv")
    array_dfs = []
    for csv in array_result_csvs:
        this_df = pd.read_csv(csv, dtype={"tgt_pdb": str}).sort_values(f"{SCORE_NAME}_score", ascending=False).drop_duplicates("lig_pdb")
        array_dfs.append(this_df)
    score_df = pd.concat(array_dfs)

    os.makedirs(f"./results_casf_screening/{SCORE_NAME}_preds", exist_ok=True)
    for target_name, df in score_df.groupby("tgt_pdb"):
        df["#code_ligand_num"] = df["lig_pdb"].map(lambda s: f"{s}_ligand_0")
        df_score = df[["#code_ligand_num", f"{SCORE_NAME}_score"]].rename({f"{SCORE_NAME}_score": "score"}, axis=1)
        df_score.to_csv(f"./results_casf_screening/{SCORE_NAME}_preds/{target_name}_score.dat", sep=" ", index=False)
    
    res_score = calc_screening_score("./results_casf_screening/", f"{SCORE_NAME}_preds", f"{SCORE_NAME}_preds")
    with open(f"./screening_{SCORE_NAME}.yaml", "w") as f:
        yaml.safe_dump(res_score, f)

if __name__ == "__main__":
    casf_score_rank()
    casf_screen()
