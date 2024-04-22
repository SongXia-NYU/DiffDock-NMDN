from collections import defaultdict
from glob import glob
import json
import pandas as pd
import os.path as osp

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", 
                   "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]

CUR_DIR = osp.abspath(osp.dirname(__file__))

class GenScoreLitPcbaScreen(LIT_PCBA_Screen):
    def __init__(self, real_score_name, target: str, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        self.target = target
        self.real_score_name = real_score_name
        super().__init__(None, ds_root, save_root, score_col_name, **kwargs)
    
    def load_best_score_df(self):
        csv = osp.join(CUR_DIR, "preds", f"{self.target}.csv")
        if osp.exists(csv):
            best_score_df = pd.read_csv(csv)
        else:
            csvs = glob(osp.join(CUR_DIR, "preds", f"{self.target}", "*.csv"))
            dfs = [pd.read_csv(csv) for csv in csvs]
            best_score_df = pd.concat(dfs)
        rename_dict = {self.real_score_name: "best_score"}
        best_score_df = best_score_df.rename(rename_dict, axis=1)
        best_score_df["og_id"] = best_score_df["fl"].map(lambda s: int(s.split("_")[0]))
        return best_score_df

def calc_tgt(tgt: str):
    for score_name in ["vina_score", "ad4_scoring_score", "vinardo_score"]:
        calc = GenScoreLitPcbaScreen(score_name, tgt, 
                                    f"/vast/sx801/geometries/LIT-PCBA-DiffDock/{tgt}", 
                                    osp.join(CUR_DIR, "scores", score_name),
                                    score_col_name=tgt)
        calc.run()

def main():
    for tgt in TARGETS:
        calc_tgt(tgt)
    for score_name in ["vina_score", "ad4_scoring_score", "vinardo_score"]:
        sum_info = defaultdict(lambda: [])
        for tgt in TARGETS:
            sum_info["target"].append(tgt)
            with open(f"{CUR_DIR}/scores/{score_name}/screening_{tgt}.json") as f:
                info = json.load(f)
            sum_info["EF1"].append(info["EF1"])
        sum_df = pd.DataFrame(sum_info)
        sum_df.to_csv(osp.join(CUR_DIR, f"lit-pcba-sum-{score_name}.csv"), index=False)
        sum_df.to_excel(osp.join(CUR_DIR, f"lit-pcba-sum-{score_name}.xlsx"), index=False, float_format="%.2f")

if __name__ == "__main__":
    main()
