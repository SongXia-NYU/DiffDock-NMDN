from collections import defaultdict
import json
import pandas as pd
import os.path as osp

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen

TARGETS = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", 
                   "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]

CUR_DIR = osp.abspath(osp.dirname(__file__))

class GenScoreLitPcbaScreen(LIT_PCBA_Screen):
    def __init__(self, target: str, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        self.target = target
        super().__init__(None, ds_root, save_root, score_col_name, **kwargs)
    
    def load_best_score_df(self):
        csv = osp.join(CUR_DIR, "preds", f"{self.target}.csv")
        rename_dict = {"score-nmdn-pose.exp_pl_534": "best_score"}
        best_score_df = pd.read_csv(csv).rename(rename_dict, axis=1)
        best_score_df["og_id"] = best_score_df["fl"].map(lambda s: int(s.split("_")[0]))
        return best_score_df

def calc_tgt(tgt: str):
    calc = GenScoreLitPcbaScreen(tgt, f"/vast/sx801/geometries/LIT-PCBA-DiffDock/{tgt}", 
                                 osp.join(CUR_DIR, "scores"),
                                 score_col_name=tgt)
    calc.run()

def main():
    for tgt in TARGETS:
        calc_tgt(tgt)
    sum_info = defaultdict(lambda: [])
    for tgt in TARGETS:
        sum_info["target"].append(tgt)
        with open(f"{CUR_DIR}/scores/screening_{tgt}.json") as f:
            info = json.load(f)
        sum_info["EF1"].append(info["EF1"])
    sum_df = pd.DataFrame(sum_info)
    sum_df.to_csv(osp.join(CUR_DIR, "lit-pcba-sum.csv"), index=False)
    sum_df.to_excel(osp.join(CUR_DIR, "lit-pcba-sum.xlsx"), index=False, float_format="%.2f")

if __name__ == "__main__":
    main()
