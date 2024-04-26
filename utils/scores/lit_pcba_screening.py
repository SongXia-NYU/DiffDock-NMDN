import logging
import os
import os.path as osp
import json
import subprocess
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import math

from abc import ABC, abstractclassmethod
from collections import defaultdict
from glob import glob
import torch
from tqdm import tqdm
from utils.LossFn import pKd2deltaG
from tqdm.contrib.concurrent import process_map

from utils.eval.trained_folder import TrainedFolder

class LIT_PCBA_ScreeningWrapper(TrainedFolder):
    def __init__(self, train_dir: str, tgt_name: str) -> None:
        super().__init__(train_dir)
        self.train_dir = train_dir
        self.tgt_name = tgt_name
        self.diffdock: bool = tgt_name.endswith("-diffdock")
        behaviour_mapper = defaultdict(lambda: "chunks")
        behaviour_mapper["esr1_ant"] = "single"
        self.tgt_behaviour = behaviour_mapper[tgt_name.lower()]

        self._src_dir = None
        self.dst_dir = osp.join(train_dir, "lit-pcba-scores", tgt_name.lower())
        os.makedirs(self.dst_dir, exist_ok=True)

    def run(self):
        tgt_name_strip = self.tgt_name.split("-diffdock")[0]
        DROOT = f"/LIT-PCBA/{tgt_name_strip}"
        score_csv = osp.join(self.dst_dir, "score.csv")
        if not osp.exists(score_csv):
            self.load_score_info()

        # pKd only
        calc_cls = DiffDockLIT_PCBA_Screen if self.diffdock else LIT_PCBA_Screen
        pkd_calc = calc_cls(score_csv, DROOT, self.dst_dir, "score")
        pkd_calc.run()

        # we only want to read the columns
        score_df = pd.read_csv(score_csv, nrows=1)
        if "score_mdn" not in score_df.columns: return self.cleanup_csv()

        # NMDN only
        calc_cls = DiffDockLIT_PCBA_Screen if self.diffdock else LIT_PCBA_Screen
        nmdn_calc = calc_cls(score_csv, DROOT, self.dst_dir, "score_MDN_LOGSUM_DIST2_REFDIST2")
        nmdn_calc.run()
        
        # NMDN + pKd
        mdn_col_name = "score_MDN_LOGSUM_DIST2_REFDIST2"
        dst_dir = osp.join(self.dst_dir, mdn_col_name)
        os.makedirs(dst_dir, exist_ok=True)
        calc_cls = DiffDockLIT_PCBA_MixedMDNScreen if self.diffdock else LIT_PCBA_MixedMDNScreen
        pkd_calc = calc_cls(score_csv, DROOT, dst_dir, n_mdn_lig=None, n_mdn_pose=1, mdn_col_name=mdn_col_name)
        pkd_calc.run()

        # half and half screening
        pkd_best_score_df: pd.DataFrame = pkd_calc.load_best_score_df().sort_values("best_score", ascending=False)
        nmdn_best_score_df: pd.DataFrame = nmdn_calc.load_best_score_df().sort_values("best_score", ascending=False)
        n_interested = math.ceil(pkd_best_score_df.shape[0] * pkd_calc.alpha)
        n_pkd = n_interested // 2
        n_nmdn = n_interested - n_pkd
        pkd_positive_df = pkd_best_score_df[["og_id", "best_score"]].iloc[:n_pkd]
        nmdn_positive_df = nmdn_best_score_df[["og_id", "best_score"]].iloc[:n_nmdn]
        n_fake: int = pkd_best_score_df.shape[0] - n_interested
        fake_df = pd.DataFrame({"og_id": [0] * n_fake, "best_score": [-np.inf] * n_fake})
        best_score_df = pd.concat([pkd_positive_df, nmdn_positive_df, fake_df], axis=0)
        dst_dir = osp.join(self.dst_dir, "pkd_half_nmdn")
        half_half_calc = LIT_PCBA_Screen(None, DROOT, dst_dir)
        half_half_calc.run(best_score_df)

        # only run for PL422
        if osp.basename(self.cfg["folder_prefix"]) != "exp_pl_422": return self.cleanup_csv()

        for mdn_col_name in ["score_MDN_SUM_DIST2_REF", "score_MDN_LOGSUM_REF", "score_MDN_LOGSUM_DIST2_REFDIST2"]:
            dst_dir = osp.join(self.dst_dir, mdn_col_name)
            os.makedirs(dst_dir, exist_ok=True)
            for n_mdn_lig in [0.1, 0.05, 0.04, 0.03, 0.02]:
                # Mixed MDN prob and pKd
                calc_cls = DiffDockLIT_PCBA_MixedMDNScreen if self.diffdock else LIT_PCBA_MixedMDNScreen
                calc = calc_cls(score_csv, DROOT, dst_dir, n_mdn_lig=n_mdn_lig, n_mdn_pose=None, mdn_col_name=mdn_col_name)
                calc.run()
            calc = calc_cls(score_csv, DROOT, dst_dir, n_mdn_lig=None, n_mdn_pose=1, mdn_col_name=mdn_col_name)
            calc.run()
        self.cleanup_csv()

    def cleanup_csv(self) -> None:
        # clean up the score CSV files
        # best_score_csv
        subprocess.run("tar caf best_score_csv.tar.gz best_score*.csv", shell=True, check=True, cwd=self.dst_dir)
        subprocess.run("rm best_score*.csv", shell=True, check=True, cwd=self.dst_dir)
        # rank_csv
        subprocess.run("tar caf rank_csv.tar.gz rank*.csv", shell=True, check=True, cwd=self.dst_dir)
        subprocess.run("rm rank*.csv", shell=True, check=True, cwd=self.dst_dir)
        # score.csv
        subprocess.run("tar caf score_csv.tar.gz score.csv", shell=True, check=True, cwd=self.dst_dir)
        subprocess.run("rm score.csv", shell=True, check=True, cwd=self.dst_dir)

    @property
    def src_dir(self):
        if self._src_dir is None:
            possible_folders = glob(osp.join(self.train_dir, f"exp_*_test_on_{self.tgt_name.lower()}_*"))
            assert len(possible_folders) == 1, possible_folders
            self._src_dir = possible_folders[0]
        return self._src_dir
    
    def load_score_info(self):
        score_csv = osp.join(self.dst_dir, "score.csv")
        compressed_score_csv = osp.join(self.dst_dir, "score_csv.tar.gz")
        if osp.exists(compressed_score_csv):
            subprocess.run("tar xaf score_csv.tar.gz", shell=True, check=True, cwd=self.dst_dir)
            subprocess.run("rm score_csv.tar.gz", shell=True, check=True, cwd=self.dst_dir)
            return

        def _raw_pred2score(pred: torch.Tensor):
            if self.cfg["auto_pl_water_ref"]:
                return pred[:, -1].cpu().numpy().reshape(-1) / pKd2deltaG
            else:
                return pred.cpu().numpy().reshape(-1)
        
        # single behaviour
        if self.tgt_behaviour == "single":
            possible_files = glob(osp.join(self.src_dir, "loss_*_test.pt"))
            assert len(possible_files) == 1, possible_files
            d = torch.load(possible_files[0], map_location="cpu")
            res ={"score": _raw_pred2score(d["PROP_PRED"])}
            if "PROP_PRED_MDN" in d:
                res["score_mdn"] = d["PROP_PRED_MDN"].cpu().numpy().reshape(-1)
            # external MDN scores
            for key in d:
                if not key.startswith("MDN_"):
                    continue
                res[f"score_{key}"] = d[key].cpu().numpy().reshape(-1)

            possible_files = glob(osp.join(self.src_dir, "record_name*.csv"))
            record_df = pd.read_csv(possible_files[0])
            res["lig_id"] = record_df["file_handle"].values.reshape(-1)
            score_df = pd.DataFrame(res)
            score_df.to_csv(score_csv)
            return
        
        # multiple chunks behavour
        # this happens when the test set is too large for a single run
        assert self.tgt_behaviour == "chunks", self.tgt_behaviour
        possible_files = glob(osp.join(self.src_dir, "loss_*_test.pt"))
        possible_files.sort()
        res = defaultdict(lambda: [])
        for test_f in possible_files:
            chunk_name = osp.basename(test_f).split("loss_")[-1].split("_test.pt")[0]
            record_file = osp.join(self.src_dir, f"record_name_{chunk_name}.csv")

            d = torch.load(test_f, map_location="cpu")
            score_info = {"sample_id": d["sample_id"].cpu().numpy().reshape(-1).tolist()}
            for key in d.keys():
                if key.startswith("MDN_") or key.startswith("PROP_PRED"):
                    score_info[key] = d[key].cpu().numpy().reshape(-1).tolist()
            score_df = pd.DataFrame(score_info).set_index("sample_id")
            record_df = pd.read_csv(record_file).set_index("sample_id").join(score_df)
            res["score"].extend(record_df["PROP_PRED"].values.reshape(-1).tolist())
            res["lig_id"].extend(record_df["file_handle"].values.tolist())
            if self.diffdock: res["rank"].extend(record_df["rank"].values.tolist())
            if "PROP_PRED_MDN" in d:
                res["score_mdn"].extend(d["PROP_PRED_MDN"].cpu().numpy().reshape(-1).tolist())
            # external MDN scores
            for key in d:
                if not key.startswith("MDN_"):
                    continue
                res[f"score_{key}"].extend(record_df[key].values.reshape(-1).tolist())

        score_df = pd.DataFrame(res)
        score_df.to_csv(score_csv)
        return
    
class ScreenPowerCalculator(ABC):
    def __init__(self, save_root, run_name, alpha=0.01, topk: int=None) -> None:
        # select top k percent for screening
        self.alpha: float = alpha
        self.topk: int = topk
        self.save_root = save_root
        self.run_name = run_name

    def run(self, best_score_df: Optional[pd.DataFrame] = None):
        os.makedirs(self.save_root, exist_ok=True)
        run_name = self.run_name

        active_df, inactive_df = self.load_label_dfs()
        # convert them into sets to look up in O(1) time
        active_og_ids = set(active_df["og_id"].values.tolist())
        inactive_og_ids = set(inactive_df["og_id"].values.tolist())

        best_score_df = self.load_best_score_df() if best_score_df is None else best_score_df
        best_score_df = best_score_df.sort_values(by="best_score", ascending=False)
        best_score_df["rank"] = np.arange(best_score_df.shape[0])

        # --------generate rank.csv which is purely for checking the model prediction------- #
        def _og_id2rank(og_id):
            # map the rank ranked by the model
            this_score_df = best_score_df[best_score_df["og_id"] == og_id]
            if this_score_df.shape[0] == 0:
                return np.inf
            return this_score_df["rank"].item()
        active_df["model_rank"] = active_df["og_id"].map(_og_id2rank)
        active_df = active_df.sort_values(by="model_rank", ascending=True)
        rank_csv = osp.join(self.save_root, f"rank_{run_name}.csv")
        active_df.to_csv(rank_csv, index=False)
        
        # Total number of active ligands
        ntb_total = active_df.shape[0]
        # Total number of ligands
        n_ligs = best_score_df.shape[0]
        alpha = self.alpha
        n_interested = math.ceil(n_ligs * alpha)
        if self.topk is not None:
            n_interested = self.topk
            alpha = 1.0 * n_interested / n_ligs
        selected_og_ids = set(best_score_df["og_id"].values[:n_interested].tolist())

        n_true_pos = len(selected_og_ids.intersection(active_og_ids))
        n_false_pos = len(selected_og_ids.intersection(inactive_og_ids))
        assert n_true_pos + n_false_pos == len(selected_og_ids)

        ef_alpha = 1.0 * n_true_pos / (ntb_total * alpha)
        res = {"EF1": ef_alpha, "#Selected": n_interested, "#TruePos": n_true_pos, "#Actives": ntb_total}
        res_file = osp.join(self.save_root, f"screening_{run_name}.json")
        with open(res_file, "w") as f:
            json.dump(res, f, indent=2)

    @abstractclassmethod
    def load_label_dfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # required columns:
        # og_id: the ligand id
        raise not NotImplementedError

    @abstractclassmethod
    def load_best_score_df(self) -> pd.DataFrame:
        # required columns:
        # og_id: the ligand id
        # best_score: best predicted score for the ligand
        raise not NotImplementedError

class LIT_PCBA_Screen(ScreenPowerCalculator):
    def __init__(self, score_csv, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        super().__init__(save_root, score_col_name, **kwargs)
        self.score_csv = score_csv
        self.ds_root = ds_root
        self.save_root = save_root
        self.score_col_name = score_col_name

    def load_label_dfs(self):
        return self.load_lit_pcba_label_dfs(self.ds_root)
    
    @staticmethod
    def load_lit_pcba_label_dfs(ds_root):
        """
        Load Label DataFrames: active and inactive. Each DataFrame should contain "og_id" which is the unique id of the ligand in the dataset.
        """
        # ----read csv files---- #
        # tha active.smi and inactive.smi are two files provided along with LIT-PCBA dataset
        active_smi = osp.join(ds_root, "actives.smi")
        inactive_smi = osp.join(ds_root, "inactives.smi")
        active_df = pd.read_csv(active_smi, header=None, names=["smiles", "og_id"], sep="\s+")
        inactive_df = pd.read_csv(inactive_smi, header=None, names=["smiles", "og_id"], sep="\s+")
        return active_df, inactive_df

    def load_best_score_df(self):
        """
        Load a test result DataFrame that contains "og_id" and "best_score". The score should be the higher the better.
        """
        best_score_csv = osp.join(self.save_root, f"best_score_{self.score_col_name}.csv")
        if osp.exists(best_score_csv):
            return pd.read_csv(best_score_csv)
        
        score_df = pd.read_csv(self.score_csv)
        score_df["chao_id"] = score_df["lig_id"].map(self.fl2chao_id)
        # mapping_df.csv is a file that maps the geometry files generated by Chao Yang to the original smi files
        map_csv = osp.join(self.ds_root, "mapping_df.csv")
        map_df = pd.read_csv(map_csv)
        all_chao_ids = set(score_df["chao_id"].values.tolist())
        best_score_df = defaultdict(lambda: [])
        for chao_id in tqdm(all_chao_ids, desc="prepare best scores"):
            try:
                # Hakuna Matata
                og_id = map_df[map_df["chao_id"]==chao_id]["og_id"].item()
            except ValueError:
                continue
            this_score_df = score_df[score_df["chao_id"] == chao_id]
            this_best_score = np.max(this_score_df[self.score_col_name].values)

            best_score_df["chao_id"].append(chao_id)
            best_score_df["best_score"].append(this_best_score)
            best_score_df["og_id"].append(og_id)

        best_score_df = pd.DataFrame(best_score_df).sort_values(by="best_score", ascending=False)
        best_score_df["rank"] = np.arange(best_score_df.shape[0])
        best_score_df.to_csv(best_score_csv, index=False)
        return best_score_df
    
    @staticmethod
    def fl2chao_id(fl: str):
        return int(fl.split("_")[0])

class DiffDockLIT_PCBA_Screen(LIT_PCBA_Screen):
    def __init__(self, score_csv, ds_root, save_root, score_col_name="score", **kwargs) -> None:
        super().__init__(score_csv, ds_root, save_root, score_col_name, **kwargs)

    def load_best_score_df(self, rank1_only: bool=False):
        sname = self.score_col_name
        if rank1_only:
            sname += "_rank1"
        best_score_csv = osp.join(self.save_root, f"best_score_{sname}.csv")
        if osp.exists(best_score_csv):
            return pd.read_csv(best_score_csv)
        score_df = pd.read_csv(self.score_csv)
        score_df["og_id"] = score_df["lig_id"].map(self.fl2og_id)
        if rank1_only:
            score_df["is_rank1"] = score_df["rank"] == 1
            assert score_df["is_rank1"].sum() > 1 ,score_df["is_rank1"].sum()
            score_df[self.score_col_name] = score_df[self.score_col_name].where(score_df["is_rank1"], -np.inf)
        score_df = score_df[["og_id", self.score_col_name]]

        best_score_df = score_df.groupby("og_id").max().reset_index()
        best_score_df = best_score_df.rename({self.score_col_name: "best_score"}, axis=1)
        best_score_df.to_csv(best_score_csv, index=False)
        return best_score_df
    
    @staticmethod
    def is_rank1(fl: str):
        return osp.basename(fl).startswith("rank1_")

    @staticmethod
    def fl2og_id(fl: str):
        return int(fl.split("_")[0])

    def run(self):
        super().run()
        rank1_df = self.load_best_score_df(rank1_only=True)
        self.run_name += "_rank1"
        return super().run(rank1_df) 
    
class LIT_PCBA_MixedMDNScreen(ScreenPowerCalculator):
    def __init__(self, score_csv, ds_root, save_root, n_mdn_lig=None, n_mdn_pose=None, mdn_col_name: str = "score_mdn", **kwargs) -> None:
        run_name = f"mixed-lig_{n_mdn_lig}-pose_{n_mdn_pose}"
        super().__init__(save_root, run_name, **kwargs)
        self.score_csv = score_csv
        self.ds_root = ds_root
        self.save_root = save_root
        self.run_name = run_name
        self.n_mdn_lig = n_mdn_lig
        self.n_mdn_pose = n_mdn_pose
        self.mdn_col_name = mdn_col_name

    def load_label_dfs(self):
        return LIT_PCBA_Screen.load_lit_pcba_label_dfs(self.ds_root)
    
    def load_best_score_df(self):
        """
        Load a test result DataFrame that contains "og_id" and "best_score". The score should be the higher the better.
        """
        best_score_csv = osp.join(self.save_root, f"best_score_{self.run_name}.csv")
        if osp.exists(best_score_csv):
            return pd.read_csv(best_score_csv)
        
        score_df = pd.read_csv(self.score_csv)
        score_df["chao_id"] = score_df["lig_id"].map(LIT_PCBA_Screen.fl2chao_id)
        # mapping_df.csv is a file that maps the geometry files generated by Chao Yang to the original smi files
        map_csv = osp.join(self.ds_root, "mapping_df.csv")
        map_df = pd.read_csv(map_csv)
        all_chao_ids = set(score_df["chao_id"].values.tolist())
        best_score_df = defaultdict(lambda: [])
        for chao_id in tqdm(all_chao_ids, desc="prepare best scores"):
            try:
                # Hakuna Matata
                og_id = map_df[map_df["chao_id"]==chao_id]["og_id"].item()
            except ValueError:
                continue
            this_score_df = score_df[score_df["chao_id"] == chao_id].reset_index()

            if self.n_mdn_pose is not None:
                # only keep top n_mdn_pose poses for best score calculation
                decend_mdn = np.argsort(-this_score_df[self.mdn_col_name])
                this_score_df = this_score_df[decend_mdn[:self.n_mdn_pose]]

            this_best_score = np.max(this_score_df["score"].values)
            this_best_mdn = np.max(this_score_df[self.mdn_col_name].values)

            best_score_df["chao_id"].append(chao_id)
            best_score_df["best_score"].append(this_best_score)
            best_score_df["best_mdn"].append(this_best_mdn)
            best_score_df["og_id"].append(og_id)

        best_score_df = pd.DataFrame(best_score_df).sort_values(by="best_mdn", ascending=False).reset_index()
        if self.n_mdn_lig is not None:
            n_keep = self.n_mdn_lig
            if isinstance(n_keep, float):
                n_keep = math.ceil(best_score_df.shape[0]*n_keep)
            assert isinstance(n_keep, int), n_keep
            # only keep top n_mdn_lig ligands, use pKd to predict the rest.
            best_score_df.loc[n_keep:, "best_score"] = -np.inf
        best_score_df = best_score_df.sort_values(by="best_score", ascending=False)
        best_score_df["rank"] = np.arange(best_score_df.shape[0])
        best_score_df.to_csv(best_score_csv, index=False)
        return best_score_df

class DiffDockLIT_PCBA_MixedMDNScreen(LIT_PCBA_MixedMDNScreen):
    def __init__(self, score_csv, ds_root, save_root, n_mdn_lig=None, n_mdn_pose=None, mdn_col_name: str = "score_mdn", **kwargs) -> None:
        super().__init__(score_csv, ds_root, save_root, n_mdn_lig, n_mdn_pose, mdn_col_name, **kwargs)

    def load_best_score_df(self):
        """
        Load a test result DataFrame that contains "og_id" and "best_score". The score should be the higher the better.
        """
        best_score_csv = osp.join(self.save_root, f"best_score_{self.run_name}.csv")
        if osp.exists(best_score_csv):
            return pd.read_csv(best_score_csv)
        
        score_df = pd.read_csv(self.score_csv)
        score_df["og_id"] = score_df["lig_id"].map(DiffDockLIT_PCBA_Screen.fl2og_id)

        print("Computing best score df...")
        # how many poses selected are selected by NMDN score
        # does not matter anymore since we will no longer use pKd score to select poses
        if self.n_mdn_pose is not None: assert self.n_mdn_pose == 1, self.n_mdn_pose
        best_score_df = score_df.sort_values(by=self.mdn_col_name, ascending=False).drop_duplicates("og_id").reset_index()
        best_score_df = best_score_df.rename({self.mdn_col_name: "best_mdn", "score": "best_score"}, axis=1)
        if self.n_mdn_lig is not None:
            n_keep = self.n_mdn_lig
            if isinstance(n_keep, float):
                n_keep = math.ceil(best_score_df.shape[0]*n_keep)
            assert isinstance(n_keep, int), n_keep
            # only keep top n_mdn_lig ligands, use pKd to predict the rest.
            best_score_df.loc[n_keep:, "best_score"] = -np.inf
        best_score_df = best_score_df.sort_values(by="best_score", ascending=False)
        best_score_df["rank"] = np.arange(best_score_df.shape[0])
        best_score_df.to_csv(best_score_csv, index=False)
        return best_score_df

def main():
    DROOT = "/LIT-PCBA/ESR1_ant"
    score_csv = osp.join(DROOT, "linf9_xgb_scores.csv")
    save_root = osp.join(DROOT, "screening_result")
    calc = LIT_PCBA_Screen(score_csv, DROOT, save_root, "linf9_score")
    calc.run()
    calc = LIT_PCBA_Screen(score_csv, DROOT, save_root, "xgb_score")
    calc.run()

if __name__ == "__main__":
    main()
