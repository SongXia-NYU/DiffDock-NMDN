from typing import Dict
import pandas as pd
import yaml
import os.path as osp
from glob import glob

from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader
from geometry_processors.post_analysis import plot_scatter_info
from utils.scores.casf_blind_scores import score_rank_power
from utils.scores.casf_scores import calc_screening_score
from utils.scores.merck_fep_scores import MerckFEPScoreCalculator

def casf_score_rank():
    for pose_select in ["nmdn_out", "crystal_out", "diffdock_out"]:
        out_df = pd.read_csv(f"./casf_opt/{pose_select}.csv").rename({"pdb": "pdb_id"}, axis=1)
        for score_name in ["xgb_score", "linf9_score"]:
            out_df["score"] = out_df[score_name]
            scores = score_rank_power(out_df, "./casf_opt", f"{pose_select}.{score_name}.png")
            with open(f"./casf_opt/{pose_select}.{score_name}.yaml", "w") as f:
                yaml.safe_dump(scores, f)

def casf_score_rank_planb():
    out_df = pd.read_csv(f"./casf_opt/nmdn_planb_out.csv").rename({"pdb": "pdb_id"}, axis=1)
    for score_name in ["linf9_score"]:
        out_df["score"] = out_df[score_name]
        scores = score_rank_power(out_df, "./casf_opt", f"nmdn_planb_out.{score_name}.png")
        with open(f"./casf_opt/nmdn_planb_out.{score_name}.yaml", "w") as f:
            yaml.safe_dump(scores, f)

def merck_rank():
    for pose_select in ["opt_nmdn_out"]:
        out_df = pd.read_csv(f"merck-fep/{pose_select}.csv")
        rank_info_by_score, count_info_by_target = MerckFEPScoreCalculator.compute_rank_info(out_df, ["xgb_score", "linf9_score"])
        with open(osp.join("merck-fep", f"rank_info_by_score.{pose_select}.yaml"), "w") as f:
            yaml.safe_dump(rank_info_by_score, f)
        
        correct_order = "hif2a	pfkfb3	eg5	cdk8	shp2	syk	cmet	tnks2 avg".split()
        for score in rank_info_by_score:
            rank_info: Dict[str, float] = rank_info_by_score[score]
            avg_rank_info = sum([rank_info[target] * count_info_by_target[target] for target in rank_info.keys()])\
                  / sum([count_info_by_target[target] for target in count_info_by_target.keys()])
            rank_info["avg"] = avg_rank_info
            rank_df = pd.DataFrame(rank_info, index=[0]).loc[:, correct_order]
            rank_df.to_csv(osp.join("merck-fep", f"{pose_select}.{score}.csv"))
            rank_df.to_excel(osp.join("merck-fep", f"{pose_select}.{score}.xlsx"), float_format="%.2f")

def casf_screen():
    calc_screening_score
    array_result_csvs = glob("./casf_opt/screening/arrays/array.*.csv")
    array_dfs = []
    for csv in array_result_csvs:
        array_dfs.append(pd.read_csv(csv))
    score_df = pd.concat(array_dfs)

    for target_name, df in score_df.groupby("tgt_pdb"):
        df["#code_ligand_num"] = df["lig_pdb"].map(lambda s: f"{s}_ligand_0")
        df_xgb = df[["#code_ligand_num", "nmdn_xgb_score"]].rename({"nmdn_xgb_score": "score"}, axis=1)
        df_xgb.to_csv(f"./casf_opt/screening/xgb_preds/{target_name}_score.dat", sep=" ", index=False)
        df_linf9 = df[["#code_ligand_num", "nmdn_linf9_score"]].rename({"nmdn_linf9_score": "score"}, axis=1)
        df_linf9.to_csv(f"./casf_opt/screening/linf9_preds/{target_name}_score.dat", sep=" ", index=False)
    
    res_xgb = calc_screening_score("./casf_opt/screening/", "xgb_preds", "xgb_preds")
    res_linf9 = calc_screening_score("./casf_opt/screening/", "linf9_preds", "linf9_preds")
    with open("./casf_opt/screening/xgb_scores.yaml", "w") as f:
        yaml.safe_dump(res_xgb, f)
    with open("./casf_opt/screening/linf9_scores.yaml", "w") as f:
        yaml.safe_dump(res_linf9, f)

if __name__ == "__main__":
    casf_score_rank_planb()
