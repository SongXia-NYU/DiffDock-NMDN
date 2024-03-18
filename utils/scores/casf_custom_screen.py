from typing import Dict, Optional
from utils.scores.casf_blind_scores import CASFBlindScreenScore
from utils.scores.lit_pcba_screening import ScreenPowerCalculator


import numpy as np
import pandas as pd


import os
import os.path as osp
from collections import defaultdict
from glob import glob
import json

from utils.utils_functions import lazy_property

# Compute custom screening performance such as top k screening power.
# TODO: current stage: finish one score custom screening
# TODO: mixed MDN and pKd score
# TODO: save folder by topk or alpha number
# TODO: summarize mean EF
# TODO: Compare with previous method
class CASF_CustomScreen(CASFBlindScreenScore):
    def __init__(self, folder_name, cfg: dict):
        super().__init__(folder_name, cfg)

    @lazy_property
    def save_root(self):
        test_dir = osp.join(self.folder_name, "casf-custom-screen-scores")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def run(self) -> None:
        score_df_by_pdb: Dict[str, pd.DataFrame] = {pdb: df for pdb, df in self.score_df.groupby("tgt_pdb")}
        def run_single_tgt(topk: int, mdn_sel: Optional[int]):
            for pdb in score_df_by_pdb:
                single_tgt_calculator = CASF_ScreenSingleTgt(mdn_sel, self.save_root, pdb, score_df_by_pdb[pdb], topk=topk)
                single_tgt_calculator.run()
        for topk in range(5, 290, 5):
            run_single_tgt(topk, None)
            run_single_tgt(topk, 285)
        tgt_summary = defaultdict(lambda: [])
        for res_json in glob(osp.join(self.save_root, "scores", "top*", "mdn_sel*", "screening_????.json")):
            parent: str = osp.dirname(res_json)
            mdn_sel: str = osp.basename(parent)
            topk: str = osp.basename(osp.dirname(parent))
            tgt_summary["topk"].append(topk)
            tgt_summary["mdn_sel"].append(mdn_sel)
            pdb_id: str = osp.basename(res_json).split("_")[-1].split(".json")[0]
            tgt_summary["pdb_id"].append(pdb_id)
            with open(res_json) as f:
                info: dict = json.load(f)
            for key in info:
                tgt_summary[key].append(info[key])
        tgt_df = pd.DataFrame(tgt_summary).rename({"EF1": "EF"}, axis=1)
        tgt_df.to_csv(osp.join(self.save_root, "target_summary.csv"), index=False)
        
        avg_df = tgt_df.groupby(by=["topk", "mdn_sel"]).mean()
        avg_df.to_csv(osp.join(self.save_root, "performance_summay.csv"), index=True)

class CASF_ScreenSingleTgt(ScreenPowerCalculator):
    def __init__(self, mdn_select: Optional[int], run_dir, pdb, score_df: pd.DataFrame, **kwargs) -> None:
        self.pdb = pdb
        self.score_df = score_df
        # figure out save name
        sel_name: str = mdn_select if mdn_select is not None else "All"
        save_root = osp.join(run_dir, "scores", f"top{kwargs['topk']}", f"mdn_sel{sel_name}")
        self.mdn_select: Optional[int] = mdn_select
        super().__init__(save_root, pdb, **kwargs)

    def run(self):
        res_file = osp.join(self.save_root, f"screening_{self.run_name}.json")
        if osp.exists(res_file): return

        os.makedirs(self.save_root, exist_ok=True)
        return super().run()
    
    def load_label_dfs(self):
        return CASF_FromDat.load_casf_label_dfs(self.pdb)
    
    def load_best_score_df(self):
        score_df = self.score_df
        score_df["lig_pdb"] = score_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
        best_score_df = score_df.sort_values(by="MDN_SUM_DIST2_REF9.0", ascending=False).drop_duplicates("lig_pdb")
        best_score_df = best_score_df.rename({"MDN_SUM_DIST2_REF9.0": "best_score", "lig_pdb": "og_id"}, axis=1)
        if self.mdn_select is None: return best_score_df
        
        # Use NMDN score to select N ligands, and use pKd to select the rest K ligands.
        assert self.mdn_select >= self.topk, f"{self.mdn_select} <= {self.topk}"
        best_score_df = best_score_df.drop("best_score", axis=1)
        best_score_df = best_score_df.rename({"PROP_PRED": "best_score"}, axis=1).reset_index()
        if self.mdn_select >= best_score_df.shape[0]: return best_score_df
        best_score_df.loc[self.mdn_select:, "best_score"] = -np.inf
        return best_score_df
        
class CASF_FromDat(ScreenPowerCalculator):
    def __init__(self, run_dir, pdb, **kwargs) -> None:
        self.pdb = pdb
        save_root = osp.join(run_dir, "my-casf-scores")
        super().__init__(save_root, pdb, **kwargs)
        self.score_root = osp.join(run_dir, "casf-scores", "screening", "pred_data")
        self.pred_dat = osp.join(self.score_root, f"{pdb}_score.dat")

    def run(self):
        res_file = osp.join(self.save_root, f"screening_{self.run_name}.json")
        if osp.exists(res_file):
            return

        os.makedirs(self.save_root, exist_ok=True)
        return super().run()
    
    def load_label_dfs(self):
        return CASF_FromDat.load_casf_label_dfs(self.pdb)

    @staticmethod
    def load_casf_label_dfs(pdb: str):
        DROOT = "/CASF-2016-cyang"
        core_pdbs = glob(f"{DROOT}/coreset/*")
        core_pdbs = [osp.basename(f) for f in core_pdbs]
        core_pdbs = set(core_pdbs)

        TARGET_INFO = f"{DROOT}/power_screening/TargetInfo.dat"
        tgt_df = pd.read_csv(TARGET_INFO, sep="\s+").set_index("#T")
        active_ligs = set(tgt_df.loc[pdb].values.tolist())
        if np.nan in active_ligs:
            active_ligs.remove(np.nan)
        inactive_ligs = core_pdbs.difference(active_ligs)

        active_df = pd.DataFrame({"og_id": list(active_ligs)})
        inactive_df = pd.DataFrame({"og_id": list(inactive_ligs)})
        return active_df, inactive_df

    def load_best_score_df(self):
        pred_df = pd.read_csv(self.pred_dat, sep="\s+")

        def lig2pdb(s: str):
            return s.split("_")[0]

        pred_df["og_id"] = pred_df["#code_ligand_num"].map(lig2pdb)
        og_ids = set(pred_df["og_id"].values.tolist())
        best_score_info = defaultdict(lambda: [])
        for og_id in og_ids:
            this_df = pred_df[pred_df["og_id"] == og_id]
            best_score = np.max(this_df["score"].values)
            best_score_info["og_id"].append(og_id)
            best_score_info["best_score"].append(best_score)
        best_score_df = pd.DataFrame(best_score_info)
        return best_score_df

def test_casf_from_dat():
    RUN_DIR = "/scratch/sx801/scripts/physnet-dimenet1/MartiniDock/exp_pl_210_run_2023-02-07_173527__083800"
    target_dats = glob(osp.join(RUN_DIR, "casf-scores", "screening", "pred_data", "*_score.dat"))
    targets = [osp.basename(f).split("_")[0] for f in target_dats]
    for target in targets:
        calc = CASF_FromDat(RUN_DIR, target)
        calc.run()

    my_screen_info = defaultdict(lambda: [])
    my_screen_res = glob(osp.join(RUN_DIR, "my-casf-scores", "screening_*.json"))
    for res_file in my_screen_res:
        target = osp.basename(res_file).split("_")[-1].split(".json")[0]
        with open(res_file) as f:
            ef1 = json.load(f)["EF1"]
        my_screen_info["#Target"].append(target)
        my_screen_info["my_EF1"].append(ef1)
    my_screen_df = pd.DataFrame(my_screen_info).set_index("#Target")
    prev_screen_df = pd.read_csv(osp.join(RUN_DIR, "casf-scores", "screening", "model_screening_EF1.dat"), sep="\s+").set_index("#Target")

    compare_df = prev_screen_df.join(my_screen_df)
    compare_df.to_csv(osp.join(RUN_DIR, "my-casf-scores", "compare.csv"))
