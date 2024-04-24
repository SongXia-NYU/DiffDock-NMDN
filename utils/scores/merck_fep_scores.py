from collections import defaultdict
import os
import os.path as osp
from glob import glob
from typing import Dict, List, Tuple
import pandas as pd
import yaml
from scipy.stats import spearmanr

from utils.eval.TestedFolderReader import TestedFolderReader
from utils.eval.trained_folder import TrainedFolder
from utils.utils_functions import lazy_property

from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader


class MerckFEPScoreCalculator(TrainedFolder):
    def __init__(self, folder_name, cfg: dict, diffdock: bool = False):
        super().__init__(folder_name)
        self.cfg: dict = cfg
        self.diffdock: bool = diffdock
        self.save_dir = osp.join(folder_name, "merck-fep-scores")
        if self.diffdock:
            self.save_dir = osp.join(folder_name, "merck-fep-diffdock-scores")
        os.makedirs(self.save_dir, exist_ok=True)

    @lazy_property
    def scoring_reader(self):
        if self.diffdock:
            scoring_test_folder = glob(osp.join(self.folder_name, "exp_*_test_on_merck_fep-diffdock_*"))[0]
        else:
            scoring_test_folder = glob(osp.join(self.folder_name, "exp_*_test_on_merck_fep_*"))[0]
        return TestedFolderReader(scoring_test_folder)

    def run(self):
        scoring_result = self.scoring_reader.result_mapper["test"]
        # res_info = {"sample_id": scoring_result["sample_id"],
        #             "pKd_score": scoring_result["PROP_PRED"].view(-1).numpy(),
        #             "NMDN_score": scoring_result["MDN_LOGSUM_DIST2_REFDIST2"].view(-1).numpy()}
        res_info = {"sample_id": scoring_result["sample_id"].cpu(),
                    "pKd_score": scoring_result["PROP_PRED"].view(-1).cpu().numpy()}
        res_df: pd.DataFrame = pd.DataFrame(res_info).set_index("sample_id")
        record: pd.DataFrame = self.scoring_reader.only_record().set_index("sample_id")
        res_df = res_df.join(record)
        def _fl_no_rank(fl: str):
            return ".".join(fl.split(".")[:-1])
        if self.diffdock:
            res_df["file_handle"] = res_df["file_handle"].map(_fl_no_rank)
            # res_df = res_df.sort_values("NMDN_score", ascending=False).drop_duplicates("file_handle")

        # rank_info_by_score, count_info_by_target = self.compute_rank_info(res_df, ["pKd_score", "NMDN_score"])
        rank_info_by_score, count_info_by_target = self.compute_rank_info(res_df, ["pKd_score"])
        with open(osp.join(self.save_dir, "rank_info_by_score.yaml"), "w") as f:
            yaml.safe_dump(rank_info_by_score, f)
        
        correct_order = "hif2a	pfkfb3	eg5	cdk8	shp2	syk	cmet	tnks2 avg".split()
        for score in rank_info_by_score:
            rank_info: Dict[str, float] = rank_info_by_score[score]
            avg_rank_info = sum([rank_info[target] * count_info_by_target[target] for target in rank_info.keys()])\
                  / sum([count_info_by_target[target] for target in count_info_by_target.keys()])
            rank_info["avg"] = avg_rank_info
            rank_df = pd.DataFrame(rank_info, index=[0]).loc[:, correct_order]
            rank_df.to_csv(osp.join(self.save_dir, f"{score}.csv"))
            rank_df.to_excel(osp.join(self.save_dir, f"{score}.xlsx"), float_format="%.3f")

    @staticmethod
    def compute_rank_info(score_df: pd.DataFrame, score_names: List[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
        def _fl2_target(fl: str):
            target = fl.split(".")[0]
            if target == "eg5_alternativeloop":
                target = "remove"
            return target
        ds_reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")
        def _fl2exp(fl: str):
            splits = fl.split(".")
            target = splits[0]
            ligand = ".".join(splits[1:])
            if "Example" in ligand: ligand = "".join(ligand.split(" "))
            return ds_reader.query_exp_delta_g(target, ligand)
        score_df["exp"] = score_df["file_handle"].map(_fl2exp)
        score_df["target"] = score_df["file_handle"].map(_fl2_target)
        score_df = score_df[score_df["target"] != "remove"]
        # res_df columns: target, exp, pKd_score, NMDN_score
        rank_info_by_score: Dict[str, Dict[str, float]] = defaultdict(lambda: {})
        count_info_by_target: Dict[str, int] = {}
        for target, df in score_df.groupby("target"):
            for score in score_names:
                df_selected = df
                rank_info_by_score[score][target] = spearmanr(-df_selected[score], df_selected["exp"])[0].item()
            count_info_by_target[target] = df_selected.shape[0]
        rank_info_by_score = dict(rank_info_by_score)
        return rank_info_by_score, count_info_by_target

def benchmark_prime():
    ds_reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")
    n_total = 0
    weighted_sum = 0.
    for target in ds_reader.TARGETS:
        if target == "eg5_alternativeloop": continue
        prime_results = osp.join(ds_reader.ds_root, target, "results_prime.csv")
        prime_df = pd.read_csv(prime_results)
        if target == "hif2a":
            prime_df = pd.read_csv(prime_results, sep="\t")

        # Nice naming....
        pred_colname = "MMGBSA dG Bind"
        if target == "cmet": pred_colname = "MMGBSA dG Bind-1"
        exp_colname = "Exp. ΔG"
        if exp_colname not in prime_df.columns: exp_colname = "exp dg"

        spearman = spearmanr(prime_df[pred_colname].values, prime_df[exp_colname].values)[0].item()
        print(target, spearman)
        weighted_sum += spearman * prime_df.shape[0]
        n_total += prime_df.shape[0]
    print("weighted avg:", weighted_sum / n_total)

if __name__ == "__main__":
    benchmark_prime()