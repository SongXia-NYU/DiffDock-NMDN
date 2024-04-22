from copy import deepcopy
from typing import Dict
import pandas as pd
import yaml
import os.path as osp
from glob import glob
import os

from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader
from geometry_processors.post_analysis import plot_scatter_info
from utils.scores.casf_blind_scores import score_rank_power
from utils.scores.casf_scores import calc_screening_score
from utils.scores.merck_fep_scores import MerckFEPScoreCalculator

SCORES = ["vina_score", "ad4_scoring_score", "vinardo_score"]

def casf_score_rank():
    out_df = pd.read_csv(f"./preds/casf-scoring.csv").rename({"pdb": "pdb_id"}, axis=1)
    for score_name in SCORES:
        out_df["score"] = out_df[score_name]
        scores = score_rank_power(out_df, "./scores", f"casf-scoring.{score_name}.png")
        with open(f"./scores/casf-scoring-{score_name}.yaml", "w") as f:
            yaml.safe_dump(scores, f)

def merck_rank():
    out_df = pd.read_csv(f"preds/merck-fep.csv")
    rank_info_by_score, count_info_by_target = MerckFEPScoreCalculator.compute_rank_info(out_df, SCORES)
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

    reader = CASF2016BlindDocking("/")
    for score_name in SCORES:
        os.makedirs(f"./scores/casf-screening-{score_name}", exist_ok=True)
        finished_targets = []
        for target_name, df in score_df.groupby("tgt_pdb"):
            df["#code_ligand_num"] = df["lig_pdb"].map(lambda s: f"{s}_ligand_0")
            df_out = df[["#code_ligand_num", score_name]].rename({score_name: "score"}, axis=1)
            df_out.to_csv(f"./scores/casf-screening-{score_name}/{target_name}_score.dat", sep=" ", index=False)
            if target_name == "1e66":
                dummy_res = deepcopy(df_out)
                dummy_res["score"] = [0.] * dummy_res.shape[0]
            finished_targets.append(target_name)

        all_targets = set([s.split("_")[0] for s in reader.screen_pdbs])
        unfinished_targets = all_targets.difference(set(finished_targets))
        breakpoint()
        for target_name in unfinished_targets:
            dummy_res.to_csv(f"./scores/casf-screening-{score_name}/{target_name}_score.dat", sep=" ", index=False) 
    
        res = calc_screening_score("./scores/", f"casf-screening-{score_name}", f"casf-screening-{score_name}")
        with open(f"./scores/casf-screening-{score_name}.yaml", "w") as f:
            yaml.safe_dump(res, f)

if __name__ == "__main__":
    casf_screen()
