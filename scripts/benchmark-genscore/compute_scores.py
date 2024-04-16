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
    out_df = pd.read_csv(f"./preds/casf-scoring.csv").rename({"pdb": "pdb_id"}, axis=1)
    for score_name in ["gen_score"]:
        out_df["score"] = out_df[score_name]
        scores = score_rank_power(out_df, "./scores", f"casf-scoring.{score_name}.png")
        with open(f"./scores/casf-scoring-{score_name}.yaml", "w") as f:
            yaml.safe_dump(scores, f)

def merck_rank():
    out_df = pd.read_csv(f"preds/merck-fep.csv")
    out_df["gen_score"] = out_df["gen_score"].map(lambda s: float(s[1:-1]))
    rank_info_by_score, count_info_by_target = MerckFEPScoreCalculator.compute_rank_info(out_df, ["gen_score"])
    with open(osp.join("scores", f"merck-fep.yaml"), "w") as f:
        yaml.safe_dump(rank_info_by_score, f)
    
    correct_order = "hif2a	pfkfb3	eg5	cdk8	shp2	syk	cmet	tnks2 avg".split()
    for score in rank_info_by_score:
        rank_info: Dict[str, float] = rank_info_by_score[score]
        avg_rank_info = sum([rank_info[target] * count_info_by_target[target] for target in rank_info.keys()])\
                / sum([count_info_by_target[target] for target in count_info_by_target.keys()])
        rank_info["avg"] = avg_rank_info
        rank_df = pd.DataFrame(rank_info, index=[0]).loc[:, correct_order]
        rank_df.to_csv(osp.join("scores", f"merck-fep.{score}.csv"))
        rank_df.to_excel(osp.join("scores", f"merck-fep.{score}.xlsx"), float_format="%.2f")

def casf_screen():
    calc_screening_score
    array_result_csvs = glob("./preds/casf-screening/array.*.csv")
    array_dfs = []
    for csv in array_result_csvs:
        array_dfs.append(pd.read_csv(csv))
    score_df = pd.concat(array_dfs)

    for target_name, df in score_df.groupby("tgt_pdb"):
        df["#code_ligand_num"] = df["lig_pdb"].map(lambda s: f"{s}_ligand_0")
        df_out = df[["#code_ligand_num", "gen_score"]].rename({"gen_score": "score"}, axis=1)
        df_out.to_csv(f"./scores/casf-screening/{target_name}_score.dat", sep=" ", index=False)
   
    res = calc_screening_score("./scores/", "casf-screening", "casf-screening")
    with open("./scores/casf-screening.yaml", "w") as f:
        yaml.safe_dump(res, f)

if __name__ == "__main__":
    casf_screen()
