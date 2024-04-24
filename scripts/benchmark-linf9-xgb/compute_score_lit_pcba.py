from collections import defaultdict
import json
import pandas as pd
import os.path as osp
from glob import glob

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", 
                   "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]
# TARGETS.remove("ESR1_ant")
# TARGETS.remove("MTORC1")
# TARGETS.remove("PPARG")
# TARGETS.remove("TP53")

CUR_DIR = osp.abspath(osp.dirname(__file__))

class GenScoreLitPcbaScreen(LIT_PCBA_Screen):
    def __init__(self, target: str, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        self.target = target
        super().__init__(None, ds_root, save_root, score_col_name, **kwargs)
    
    def load_best_score_df(self):
        csvs = glob(osp.join(CUR_DIR, "preds", f"{self.target}", "*.csv"))
        rename_dict = {self.score_col_name.split(".")[0]: "best_score"}
        best_score_df = pd.concat([pd.read_csv(csv) for csv in csvs])
        best_score_df = best_score_df.rename(rename_dict, axis=1)
        best_score_df["og_id"] = best_score_df["fl"].map(lambda s: int(s.split("_")[0]))
        return best_score_df

def calc_tgt(tgt: str, score_name:str):
    calc = GenScoreLitPcbaScreen(tgt, f"/vast/sx801/geometries/LIT-PCBA-DiffDock/{tgt}", 
                                 osp.join(CUR_DIR, "scores"),
                                 score_col_name=f"{score_name}.{tgt}")
    calc.run()

def main():
    for score_name in ["xgb_score", "linf9_score"]:
        for tgt in TARGETS:
            calc_tgt(tgt, score_name)
        sum_info = defaultdict(lambda: [])
        for tgt in TARGETS:
            sum_info["target"].append(tgt)
            with open(f"{CUR_DIR}/scores/screening_{score_name}.{tgt}.json") as f:
                info = json.load(f)
            sum_info["EF1"].append(info["EF1"])
        sum_df = pd.DataFrame(sum_info)
        sum_df.to_csv(osp.join(CUR_DIR, f"lit-pcba-sum-{score_name}.csv"), index=False)
        sum_df.to_excel(osp.join(CUR_DIR, f"lit-pcba-sum-{score_name}.xlsx"), index=False, float_format="%.2f")

if __name__ == "__main__":
    main()
