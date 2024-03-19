import pandas as pd
import yaml

from geometry_processors.post_analysis import plot_scatter_info
from utils.scores.casf_blind_scores import score_rank_power

def casf_score_rank():
    for pose_select in ["crystal_out", "nmdn_out", "diffdock_out"]:
        out_df = pd.read_csv(f"{pose_select}.csv").rename({"pdb": "pdb_id"}, axis=1)
        for score_name in ["xgb_score", "linf9_score"]:
            out_df["score"] = out_df[score_name]
            scores = score_rank_power(out_df, "./casf", f"{pose_select}.{score_name}.png")
            with open(f"./casf/{pose_select}.{score_name}.yaml", "w") as f:
                yaml.safe_dump(scores, f)

if __name__ == "__main__":
    casf_score_rank()
