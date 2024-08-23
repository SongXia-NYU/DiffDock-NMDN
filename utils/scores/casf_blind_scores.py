"""
Blind docking and screening on CASF-2016 data set.
"""
from collections import defaultdict
import os
import os.path as osp
from typing import Dict, List, Set
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import torch
import subprocess
from omegaconf import OmegaConf

from utils.eval.TestedFolderReader import TestedFolderReader
from utils.eval.tester import Tester
from utils.scores.casf_scores import CASF_ROOT, RANKING_CSV, CasfScoreCalculator, calc_screening_score, get_rank, plot_scatter_info
from utils.utils_functions import lazy_property, option_solver
from utils.configs import read_config_file


class CASFBlindScreenScore(CasfScoreCalculator):
    def __init__(self, folder_name, cfg: dict):
        super().__init__(folder_name, cfg)

    def run(self) -> None:
        computed_keys: Set[str] = set()
        for tgt_pdb, this_df in self.score_df.groupby("tgt_pdb"):
            assert isinstance(this_df, pd.DataFrame)
            for key in this_df.columns:
                if not key.startswith(("MDN_", "PROP_")): continue
                computed_keys.add(key)
                resdir = osp.join(self.save_root, f"screening_scores_{key}")
                os.makedirs(resdir, exist_ok=True)
                df2save: pd.DataFrame = this_df[["#code_ligand_num", key]].rename({key: "score"}, axis=1)
                df2save.to_csv(osp.join(resdir, f"{tgt_pdb}_score.dat"), sep=" ", index=False)

            # MDN select pose, then pkd score screen
            this_df["lig_pdb"] = this_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
            for mdn_name in ["MDN_LOGSUM_DIST2_REFDIST2"]:
                if mdn_name not in this_df.columns:
                    continue
                computed_keys.add(f"{mdn_name}_PKd")
                pose_selected_df = this_df.sort_values(mdn_name, ascending=False).drop_duplicates("lig_pdb")
                resdir = osp.join(self.save_root, f"screening_scores_{mdn_name}_PKd")
                os.makedirs(resdir, exist_ok=True)
                df2save: pd.DataFrame = pose_selected_df[["#code_ligand_num", "PROP_PRED"]].rename({"PROP_PRED": "score"}, axis=1)
                df2save.to_csv(osp.join(resdir, f"{tgt_pdb}_score.dat"), sep=" ", index=False)
        
        score_sum_info: List[pd.DataFrame] = []
        for key in computed_keys:
            screen_scores = calc_screening_score(self.save_root, f"screening_scores_{key}", f"screening_{key}")
            this_info = pd.DataFrame(screen_scores, index=[key])
            score_sum_info.append(this_info)

        # clean up
        subprocess.run("tar caf bk_screening_scores.tar.gz screening_scores_*", shell=True, check=True, cwd=self.save_root)
        subprocess.run("rm -r screening_scores_* ", shell=True, check=True, cwd=self.save_root)
        subprocess.run("tar caf bk_model_screening_efs.tar.gz model_screening_*.dat ", shell=True, check=True, cwd=self.save_root)
        subprocess.run("rm model_screening_*.dat ", shell=True, check=True, cwd=self.save_root)
        subprocess.run("tar caf bk_screening_outs.tar.gz screening_*.out ", shell=True, check=True, cwd=self.save_root)
        subprocess.run("rm screening_*.out ", shell=True, check=True, cwd=self.save_root)
        subprocess.run("tar caf bk_screen-scores.csv.tar.gz screen-scores.csv", shell=True, check=True, cwd=self.save_root)
        subprocess.run("rm screen-scores.csv", shell=True, check=True, cwd=self.save_root)

        score_sum_df = pd.concat(score_sum_info)
        score_sum_df.to_csv(osp.join(self.save_root, "screening_summary.csv"))

    @lazy_property
    def score_df(self) -> pd.DataFrame:
        score_csv = osp.join(self.save_root, "screen-scores.csv")
        if osp.exists(score_csv): return pd.read_csv(score_csv)

        def _fl2tgt_pdb(fl: str):
            return fl.split("_")[0]
        def _fl2code_ligand(fl: str):
            lig_pdb = fl.split("_")[1].split(".")[0]
            lig_id = fl.split("lig_srcrank")[-1]
            code_lig = f"{lig_pdb}_ligand_{lig_id}"
            return code_lig
        
        score_df = self.load_test_result()
        score_df["#code_ligand_num"] = score_df["file_handle"].map(_fl2code_ligand)
        score_df["tgt_pdb"] = score_df["file_handle"].map(_fl2tgt_pdb)
        score_df.to_csv(score_csv, index=False)
        return score_df
    
    def load_test_result(self) -> pd.DataFrame:
        # multiple chunk haviour
        score_dfs = []
        for test_name in self.blind_screening_reader.result_mapper:
            # test_name: "test@part_$i"
            part_id: str = test_name.split("@")[-1]
            record_df: pd.DataFrame = self.blind_screening_reader.record_mapper[part_id].set_index("sample_id")
            n_res = record_df.shape[0]
            raw_pred_dict: dict = self.blind_screening_reader.result_mapper[test_name]

            pred_info = {}
            for key in raw_pred_dict:
                if not isinstance(raw_pred_dict[key], torch.Tensor): continue
                if raw_pred_dict[key].view(-1).shape[0] != raw_pred_dict["sample_id"].shape[0]: continue
                pred_info[key] = raw_pred_dict[key].view(-1).cpu().numpy().tolist()

            pred_df = pd.DataFrame(pred_info).set_index("sample_id")
            pred_df = record_df.join(pred_df).reset_index()
            # sample_id is temporary
            score_dfs.append(pred_df.drop("sample_id", axis=1))
        score_dfs = pd.concat(score_dfs, axis=0)
        return score_dfs


    @lazy_property
    def save_root(self):
        test_dir = osp.join(self.folder_name, "casf-blind-scores")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    @lazy_property
    def blind_screening_reader(self) -> TestedFolderReader:
        return TestedFolderReader(self.blind_screening_test_folder)
    
    @lazy_property
    def blind_screening_test_folder(self):
        name = self.screening_ds_cfg["short_name"]
        if self.ref: name += "-ref"
        return self.get_folder(name)
    
    @lazy_property
    def screening_ds_cfg(self) -> dict:
        test_ds_args: dict = Tester.parse_explicit_ds_args(self.cfg["screening_config"])
        return test_ds_args


class CASFBlindDockScore(CasfScoreCalculator):
    def __init__(self, folder_name, cfg):
        super().__init__(folder_name, cfg)

    @lazy_property
    def save_root(self):
        test_dir = osp.join(self.folder_name, "casf-blind-scores")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir
    
    def run(self):
        test_res: dict = self.blind_docking_reader.result_mapper["test"]
        test_record: pd.DataFrame = self.blind_docking_reader.only_record().set_index("sample_id")
        out_dfs: List[pd.DataFrame] = []
        # Scoring, ranking and docking power of individual scores
        for key in test_res:
            if not (key.startswith("MDN_") or key in ["PROP_PRED", "PROP_PRED_MDN"]):
                continue
            score = test_res[key].cpu().numpy().reshape(-1)
            sample_id = test_res["sample_id"].cpu().numpy()
            # required columns: file_handle, score, pdb_id
            score_df = pd.DataFrame({"sample_id": sample_id, "score": score}).set_index("sample_id").join(test_record)
            if key == "MDN_LOGSUM_DIST2_REFDIST2":
                self.docking_detailed(score_df)
            max_score_df = score_df.set_index("file_handle").join(self.rmsd_info_df, how="outer").sort_values("score", ascending=False).drop_duplicates(["pdb_id"]).dropna()
            scores = self.compute_scores(max_score_df)
            this_scores_df = pd.DataFrame(scores, index=[key])
            out_dfs.append(this_scores_df)

        for mdn_name in ["MDN_SUM_DIST2_REF", "MDN_LOGSUM_REF", "MDN_LOGSUM_DIST2_REF", "MDN_LOGSUM_DIST2_REFDIST2"]:
            self.mixed_mdn_pkd_score(out_dfs, mdn_name)

        out_df: pd.DataFrame = pd.concat(out_dfs)
        out_df.to_csv(osp.join(self.save_root, "docking_summary.csv"))
        def to3decimal(num: float):
            return "{:.3f}".format(num)
        out_df[["score_r", "rank_spearman", "rank_kendall"]] = out_df[["score_r", "rank_spearman", "rank_kendall"]].applymap(to3decimal)
        out_df.to_excel(osp.join(self.save_root, "docking_summary.xlsx"))

    def docking_detailed(self, score_df: pd.DataFrame):
        # calculate detailed docking scores like in the DiffDock work
        score_df = score_df.set_index("file_handle").join(self.rmsd_info_df, how="outer")
        score_df["rank"] = score_df.index.map(lambda s: int(s.split("srcrank")[-1]))
        n_sel = 10
        if n_sel is not None:
            np.random.seed(54)
            sel_rank = np.random.permutation(np.arange(1, 101))[:n_sel]
            score_df_sel = score_df.reset_index().set_index("rank").loc[sel_rank].reset_index().set_index("file_handle")
        out_df = self.diffdock_detailed_docking_power(score_df_sel)
       
        n_sel = "" if n_sel is None else n_sel
        out_df.to_csv(osp.join(self.save_root, f"docking_detailed{n_sel}.csv"))
        out_df.to_excel(osp.join(self.save_root, f"docking_detailed{n_sel}.xlsx"), float_format="%.2f")
        print(out_df)
    
    @staticmethod
    def diffdock_detailed_docking_power(score_df: pd.DataFrame):
        nmdn_top1_rmsd = []
        nmdn_top5_rmsd = []
        confidence_top1_rmsd = []
        confidence_top5_rmsd = []
        for pdb, df in score_df.groupby("pdb_id"):
            df = df.sort_values("score", ascending=False)
            nmdn_top1_rmsd.append(df.iloc[0]["rmsd"].item())
            nmdn_top5_rmsd.append(df.iloc[:5]["rmsd"].values.min().item())
            df = df.sort_values("rank", ascending=True)
            confidence_top1_rmsd.append(df.iloc[0]["rmsd"].item())
            confidence_top5_rmsd.append(df.iloc[:5]["rmsd"].values.min().item())
        nmdn_top1_rmsd = np.asarray(nmdn_top1_rmsd)
        nmdn_top5_rmsd = np.asarray(nmdn_top5_rmsd)
        confidence_top1_rmsd = np.asarray(confidence_top1_rmsd)
        confidence_top5_rmsd = np.asarray(confidence_top5_rmsd)
        out_info = defaultdict(lambda: [])
        out_info["top1%<2"].append(100.0 * (confidence_top1_rmsd < 2.).sum() / 285)
        out_info["top1-median"].append(np.median(confidence_top1_rmsd))
        out_info["top5%<2"].append(100.0 * (confidence_top5_rmsd < 2.).sum() / 285)
        out_info["top5-median"].append(np.median(confidence_top5_rmsd))

        out_info["top1%<2"].append(100.0 * (nmdn_top1_rmsd < 2.).sum() / 285)
        out_info["top1-median"].append(np.median(nmdn_top1_rmsd))
        out_info["top5%<2"].append(100.0 * (nmdn_top5_rmsd < 2.).sum() / 285)
        out_info["top5-median"].append(np.median(nmdn_top5_rmsd))
        out_df = pd.DataFrame(out_info, index=["DiffDock", "DiffDock-NMDN"])
        return out_df

    def mixed_mdn_pkd_score(self, out_dfs: List[pd.DataFrame], mdn_name: str):
        test_res: dict = self.blind_docking_reader.result_mapper["test"]
        if mdn_name not in test_res:
            print(f"NMDN score: {mdn_name} not found, skipping...")
            return
        test_record: pd.DataFrame = self.blind_docking_reader.only_record().set_index("sample_id")

        pkd_score = test_res["PROP_PRED"].numpy().reshape(-1)
        mdn_score = test_res[mdn_name].numpy().reshape(-1)
        sample_id = test_res["sample_id"].numpy()
        mixed_mdn_df = pd.DataFrame({"sample_id": sample_id, "score": pkd_score, "mdn_score": mdn_score}).set_index("sample_id").join(test_record)
        max_score_df = mixed_mdn_df.set_index("file_handle").join(self.rmsd_info_df, how="outer").sort_values("mdn_score", ascending=False).drop_duplicates(["pdb_id"])
        mixed_mdn_score = self.compute_scores(max_score_df)
        out_dfs.append(pd.DataFrame(mixed_mdn_score, index=[f"Mixed-{mdn_name}-pKd"]))

        # sample N geometries
        for n_sample in range(10, 21, 10):
            mixed_mdn_df_sampled = mixed_mdn_df.set_index("file_handle").join(self.rmsd_info_df, how="outer").groupby("pdb_id").sample(n=n_sample)
            max_score_df = mixed_mdn_df_sampled.sort_values("mdn_score", ascending=False).drop_duplicates(["pdb_id"])
            mixed_mdn_score = self.compute_scores(max_score_df)
            out_dfs.append(pd.DataFrame(mixed_mdn_score, index=[f"Mixed-{mdn_name}-pKd-sample{n_sample}"]))
        return

    def compute_scores(self, max_score_df: pd.DataFrame) -> dict:
        scores: dict = {}
        tgt_df = pd.read_csv(osp.join(CASF_ROOT, "CASF-2016.csv"))[["pdb", "pKd"]].set_index("pdb")
        scoring_df = max_score_df.set_index("pdb_id").join(tgt_df)

        # scoring and ranking power
        exp = scoring_df["pKd"].values
        cal = scoring_df["score"].values
        r = pearsonr(exp, cal)[0]
        scores["score_r"] = r
        rank_info: pd.DataFrame = pd.read_csv(RANKING_CSV)
        pdb = scoring_df.index.values.tolist()
        df = pd.DataFrame({"pdb": pdb, "PRED": cal.tolist(), "TGT": exp.tolist()}).set_index("pdb")
        df = pd.concat([df, rank_info.set_index("pdb")], axis=1)
        df = df.reset_index()
        spearman, kendall, __ = get_rank(df.dropna(), rank_info)
        scores["rank_spearman"] = spearman
        scores["rank_kendall"] = kendall

        # docking power without crystal structure
        n_total = 285
        n_success = (max_score_df["rmsd"] <= 2.0).sum()
        sr1 = 1.0 * n_success / n_total
        sr1 = "{:.1f}%".format(100.*sr1)
        # record docking score numbers
        scores["docking_SR1"] = sr1
        scores["docking_n_success"] = n_success
        scores["docking_n_total"] = n_total
        return scores
    
    @lazy_property
    def blind_docking_reader(self) -> TestedFolderReader:
        return TestedFolderReader(self.blind_docking_test_folder)
    
    @lazy_property
    def blind_docking_test_folder(self):
        name = self.docking_ds_cfg["short_name"]
        if self.ref: name += "-ref"
        return self.get_folder(name)
    
    @lazy_property
    def docking_ds_cfg(self) -> dict:
        test_ds_args: dict = read_config_file(self.cfg["docking_config"])
        test_ds_args = OmegaConf.merge(self.cfg, test_ds_args)
        return test_ds_args
    
    @lazy_property
    def rmsd_info_df(self):
        test_ds_args: dict = self.docking_ds_cfg
        data_root: str = test_ds_args.data.data_root
        ds_name = option_solver(test_ds_args.data.data_provider)["dataset_name"].split(".pyg")[0]
        rmsd_info_csv: str = osp.join(osp.dirname(data_root), f"{ds_name}.rmsd2crystal.csv")
        rmsd_info_df: pd.DataFrame = pd.read_csv(rmsd_info_csv).set_index("file_handle")
        return rmsd_info_df

def score_rank_power(score_df: pd.DataFrame, save_folder: str, save_name: str):
    scores: dict = {}
    tgt_df = pd.read_csv(osp.join(CASF_ROOT, "CASF-2016.csv"))[["pdb", "pKd"]].set_index("pdb")
    scoring_df = score_df.set_index("pdb_id").join(tgt_df)

    # scoring and ranking power
    exp = scoring_df["pKd"].values
    cal = scoring_df["score"].values
    plot_scatter_info(exp, cal, save_folder, save_name, "Exp vs. Cal")
    r = pearsonr(exp, cal)[0]
    scores["score_r"] = r.item()
    rank_info: pd.DataFrame = pd.read_csv(RANKING_CSV)
    pdb = scoring_df.index.values.tolist()
    df = pd.DataFrame({"pdb": pdb, "PRED": cal.tolist(), "TGT": exp.tolist()}).set_index("pdb")
    df = pd.concat([df, rank_info.set_index("pdb")], axis=1)
    df = df.reset_index()
    spearman, kendall, __ = get_rank(df.dropna(), rank_info)
    scores["rank_spearman"] = spearman.item()
    scores["rank_kendall"] = kendall.item()
    return scores
