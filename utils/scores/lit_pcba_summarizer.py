from collections import defaultdict
import os.path as osp
from glob import glob
from typing import Dict, List
import json
import pandas as pd
import subprocess

from utils.utils_functions import lazy_property

class LIT_PCBA_Summarizer:
    def __init__(self, trained_dir: str) -> None:
        self.workdir = osp.join(trained_dir, "lit-pcba-scores")

    def run(self):
        info_list: List[pd.DataFrame] = []
        for target in self.targets:
            info_list.append(self.single_target_scores(target))
        sum_df = pd.concat(info_list)
        sum_df.to_csv(osp.join(self.workdir, "sum_df.csv"), index=False)

        # reorganize dataframe
        ef_reorganize_dfs = []
        sum_df = sum_df.sort_values(by="target")
        for tgt_name, df in sum_df.groupby("target"):
            df = df.set_index("score_name")[["EF1"]].rename({"EF1": f"EF1-{tgt_name}"}, axis=1)
            ef_reorganize_dfs.append(df)
        ef_reorganize_df = pd.concat(ef_reorganize_dfs, axis=1)
        ef_reorganize_df.to_csv(osp.join(self.workdir, "ef_reorganize.csv"))
        ef_reorganize_df.to_excel(osp.join(self.workdir, "ef_reorganize.xlsx"), float_format="%.3f")

        # clean up
        for target in self.targets:
            subprocess.run(f"tar caf '{target}.tar.gz' '{target}'", cwd=self.workdir, shell=True, check=True)
            subprocess.run(f"rm -r '{target}'", cwd=self.workdir, shell=True, check=True)

    def single_target_scores(self, target: str) -> pd.DataFrame:
        tgt_info = defaultdict(lambda: [])
        tgt_dir = osp.join(self.workdir, target)
        # name: screening_$name.json
        score_json_by_name: Dict[str, str] = {}
        # regular scores
        for json_f in glob(osp.join(tgt_dir, "screening_*.json")):
            name = osp.basename(json_f).split(".json")[0].split("screening_")[-1]
            score_json_by_name[name] = json_f
        # mixed MDN scores
        for json_f in glob(osp.join(tgt_dir, "score_*", "screening_*.json")):
            name = osp.basename(json_f).split(".json")[0].split("screening_")[-1]
            mdn_score_name = osp.basename(osp.dirname(json_f))
            score_name = f"{mdn_score_name}.{name}"
            score_json_by_name[score_name] = json_f
        
        for score_name in score_json_by_name:
            score_json = score_json_by_name[score_name]
            with open(score_json) as f: scores: dict = json.load(f)
            tgt_info["target"].append(target)
            tgt_info["score_name"].append(score_name)
            for key in scores: tgt_info[key].append(scores[key])
        return pd.DataFrame(tgt_info)

    @lazy_property
    def targets(self):
        screen_score_jsons = glob(osp.join(self.workdir, "*", "screening_score.json"))
        targets = [osp.basename(osp.dirname(screen_score_json)) for screen_score_json in screen_score_jsons]
        return targets
