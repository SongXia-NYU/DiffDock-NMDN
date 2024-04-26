from collections import defaultdict
import os
import os.path as osp
import pandas as pd
from glob import glob
import json
import seaborn as sns
import matplotlib.pyplot as plt
from utils.scores.lit_pcba_screening import DiffDockLIT_PCBA_MixedMDNScreen, DiffDockLIT_PCBA_Screen, LIT_PCBA_ScreeningWrapper


class LIT_PCBA_CustomWrapper(LIT_PCBA_ScreeningWrapper):
    def __init__(self, train_dir: str, tgt_name: str) -> None:
        super().__init__(train_dir, tgt_name)
        self.dst_dir = osp.join(train_dir, "lit-pcba-custom-scores", tgt_name.lower())
        os.makedirs(self.dst_dir, exist_ok=True)

    def run(self):
        assert self.diffdock
        assert osp.basename(self.cfg["folder_prefix"]) == "exp_pl_422", self.cfg["folder_prefix"]
        tgt_name_strip = self.tgt_name.split("-diffdock")[0]
        DROOT = f"/LIT-PCBA/{tgt_name_strip}"
        score_csv = osp.join(self.dst_dir, "score.csv")
        if not osp.exists(score_csv):
            self.load_score_info()

        mdn_col_name = "score_MDN_LOGSUM_DIST2_REFDIST2"
        topk_list = [10, 20, 30, 50]
        for topk in range(100, 2001, 100): topk_list.append(topk)
        for topk in topk_list:
            sroot = osp.join(self.dst_dir, f"top{topk}")
            calc = DiffDockLIT_PCBA_Screen(score_csv, DROOT, sroot, mdn_col_name, topk=topk)
            calc.run()
            calc = DiffDockLIT_PCBA_MixedMDNScreen(score_csv, DROOT, sroot, n_mdn_lig=None, n_mdn_pose=1, mdn_col_name=mdn_col_name, topk=topk)
            calc.run()

        sum_df: pd.DataFrame = self.summarize()
        for score_name in ["mixed-lig_None-pose_1", "score_MDN_LOGSUM_DIST2_REFDIST2"]:
            this_df = sum_df[sum_df["score_name"] == score_name].set_index("topk")
            this_df = this_df.rename({"EF1": "Enhancement Factor", "#TruePos": "Number of True Positives"}, axis=1)
            plt.figure()
            sns.lineplot(this_df[["Number of True Positives"]], markers=True, dashes=False)
            ax2 = plt.twinx()
            sns.lineplot(this_df[["Enhancement Factor"]], markers=True, dashes=False, ax=ax2, palette=["orange"])
            ax2.set_ylim(top=ax2.get_ylim()[-1]+0.1)
            plt.title(self.tgt_name)
            plt.tight_layout()
            plt.savefig(osp.join(self.dst_dir, f"ef_vs_topk_{score_name}.png"))
            plt.close()
        return

    def summarize(self) -> pd.DataFrame:
        sum_info = defaultdict(lambda: [])
        topk_folders = glob(osp.join(self.dst_dir, "top*"))
        for topk_folder in topk_folders:
            topk: int = int(osp.basename(topk_folder).split("top")[-1])
            score_jsons = glob(osp.join(topk_folder, "screening_*.json"))
            for score_json in score_jsons:
                score_name: str = osp.basename(score_json).split("screening_")[-1].split(".json")[0]
                with open(score_json) as f: scores: dict = json.load(f)
                sum_info["topk"].append(topk)
                sum_info["score_name"].append(score_name)
                for key in scores: sum_info[key].append(scores[key])
        sum_df = pd.DataFrame(sum_info)
        sum_df.to_csv(osp.join(self.dst_dir, "sum_df.csv"), index=False)
        return sum_df