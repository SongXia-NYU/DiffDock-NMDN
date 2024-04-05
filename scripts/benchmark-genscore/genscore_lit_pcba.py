import pandas as pd
import os.path as osp

from utils.scores.lit_pcba_screening import LIT_PCBA_Screen


class GenScoreLitPcbaScreen(LIT_PCBA_Screen):
    def __init__(self, target: str, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        self.target = target
        super().__init__(None, ds_root, save_root, score_col_name, **kwargs)
    
    def load_best_score_df(self):
        csv = osp.join(osp.abspath(osp.dirname(__file__)), "preds", f"{self.target}.csv")
        rename_dict = {"score-nmdn-pose.exp_pl_534": "best_score"}
        best_score_df = pd.read_csv(csv).rename(rename_dict, axis=1)
        best_score_df["og_id"] = best_score_df["fl"].map(lambda s: int(s.split("_")[0]))
        return best_score_df

def calc_tgt(tgt: str):
    calc = GenScoreLitPcbaScreen(tgt, f"/vast/sx801/geometries/LIT-PCBA-DiffDock/{tgt}", 
                                 osp.join(osp.abspath(osp.dirname(__file__)), "scores"),
                                 score_col_name=tgt)
    calc.run()

def main():
    for tgt in ["ESR1_ago", "ESR1_ant", "PPARG", "TP53"]:
        calc_tgt(tgt)

if __name__ == "__main__":
    main()
