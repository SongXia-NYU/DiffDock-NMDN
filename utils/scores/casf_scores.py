import math
import os
import os.path as osp
import shutil
import subprocess
from glob import glob
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, gaussian_kde, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.LossFn import pKd2deltaG
from utils.eval.TestedFolderReader import TestedFolderReader
from utils.eval.trained_folder import TrainedFolder
from utils.utils_functions import lazy_property

CASF_ROOT = "/CASF-2016-cyang/"
RANKING_CSV = "/CASF-2016-cyang/CASF-2016.csv"
if not osp.exists("/scratch/sx801"):
    RANKING_CSV = "/home/carrot_of_rivia/Documents/PycharmProjects/CASF-2016-cyang/CASF-2016.csv"


class CasfScoreCalculator(TrainedFolder):
    @lazy_property
    def save_root(self):
        test_dir = osp.join(self.folder_name, "casf-scores")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir

    def __init__(self, folder_name, cfg: dict):
        super().__init__(folder_name)
        self.folder_name = folder_name
        for key in cfg:
            OmegaConf.update(self.cfg, key, cfg[key], force_add=True)
        self.ref = cfg["ref"]

        self._docking_test_folder = None
        self._docking_reader = None
        self._scoring_test_folder = None
        self._scoring_reader = None
        self._screening_test_folder = None
        self._screening_reader = None
        self._use_mixed_scores = None

        self.result_summary = {}

    def normal_run(self):
        scoring_df = self.scoring_ranking_score()
        self.docking_score(scoring_df)

        try:
            __ = self.screening_test_folder
            self.screening_score()
        except AssertionError:
            print("screening test folder does not exist, skipping...")
        except FileNotFoundError:
            print("screening test not complete, skipping...")
        
        self.save_summary()

        # Evaluation using only one of the predicted scores
        pred_keys: List[str] = list(self.scoring_reader.result_mapper["test"].keys())
        actions = ["default", "mdn"]
        actions.extend([key for key in pred_keys if key.startswith("MDN_")])
        for action in actions:
            try:
                scoring_df = self.scoring_ranking_score(action=action)
            except KeyError as e:
                print(e)
                continue
            self.docking_score(scoring_df, action=action)

            try:
                __ = self.screening_test_folder
                self.screening_score(action=action)
            except AssertionError:
                print("screening test folder does not exist, skipping...")
            except FileNotFoundError:
                print("screening test not complete, skipping...")
            self.save_summary(f"result_summary_{action}")

    def mixed_mdn_run(self):
        # NMDN select pose and pKd score.
        self.screening_mixed_scores(None, 1)
        # manually turned off 8/21/2023 to save computation time
        # does not work very well anyway.
        return
        try:
            __ = self.screening_test_folder
            self.screening_mixed_scores(0.1, None)
            self.screening_mixed_scores(0.05, None)
            self.screening_mixed_scores(0.04, None)
            self.screening_mixed_scores(0.03, None)
            self.screening_mixed_scores(0.02, None)
        except AssertionError:
            print("screening test folder does not exist, skipping...")
        except FileNotFoundError:
            print("screening test not complete, skipping...")

    def run(self):
        self.normal_run()
        if self.use_mixed_scores:
            self.mixed_mdn_run()
        self.cleanup()

    def cleanup(self):
        # compress .dat files to reduce the number of files on the system
        potential_folders = glob(osp.join(self.save_root, "docking*"))
        potential_folders.extend(glob(osp.join(self.save_root, "screening*")))
        potential_folders = [f for f in potential_folders if osp.isdir(f)]
        clean_names = [osp.basename(f) for f in potential_folders]

        # compress predictions
        for name in clean_names:
            compress_cmd = f"tar caf {name}.tar.gz {name}"
            subprocess.run(compress_cmd, shell=True, check=True, cwd=self.save_root)
            shutil.rmtree(osp.join(self.save_root, name))
        if clean_names:
            # move less frequently used files into an archive folder
            os.makedirs(osp.join(self.save_root, "archive"), exist_ok=True)
            subprocess.run("mv *.tar.gz archive", shell=True, check=True, cwd=self.save_root)
            subprocess.run("mv *.png archive", shell=True, check=True, cwd=self.save_root)
            subprocess.run("mv kendall_*.csv archive", shell=True, check=True, cwd=self.save_root)
            subprocess.run("mv spearman_*.csv archive", shell=True, check=True, cwd=self.save_root)

        # summarize all score performances
        score_csvs = glob(osp.join(self.save_root, "result_summary_*.csv"))
        score_csv_by_name: Dict[str, str] = {}
        for score_csv in score_csvs:
            score_name = osp.basename(score_csv).split("result_summary_")[-1].split(".csv")[0]
            score_csv_by_name[score_name] = score_csv
        score_csv_by_name["pKd"] = osp.join(self.save_root, "result_summary.csv")
        sum_score_info = []
        for score_name in score_csv_by_name:
            this_df = pd.read_csv(score_csv_by_name[score_name])
            this_df["score_name"] = [score_name]
            sum_score_info.append(this_df.set_index(["score_name"]))
        sum_score_df = pd.concat(sum_score_info)
        sum_score_df.to_csv(osp.join(self.save_root, "sum_scores.csv"))
        # save as excel for easier copy pasta
        def _three_digits(raw: float) -> str:
            return "{:.3f}".format(raw)
        sum_score_df[["score_r", "rank_spearman"]] = sum_score_df[["score_r", "rank_spearman"]].applymap(_three_digits)
        wanted_col_names: List[str] = []
        for name in ["score_r", "rank_spearman", "docking_SR1", "screening_EF1", "screening_SR1"]:
            if name in sum_score_df.columns:
                wanted_col_names.append(name)
        sum_score_df = sum_score_df[wanted_col_names]
        sum_score_df.to_excel(osp.join(self.save_root, "sum_scores.xlsx"))


    def screening_score(self, action: str="default"):
        screening_name = "screening"
        if action == "default":
            mol_prop_name = "PROP_PRED"
            if self.use_mixed_scores:
                screening_name += "-pKd"
        elif action == "mdn":
            mol_prop_name = "PROP_PRED_MDN"
            screening_name += "-mdn"
        else:
            assert action.startswith("MDN_"), action
            mol_prop_name = action
            screening_name += "-" + action
            
        if self.ref:
            screening_name += "-ref"
        os.makedirs(osp.join(self.save_root, screening_name, "pred_data"), exist_ok=True)

        # screening
        screening_result = self.screening_reader.result_mapper
        for key in screening_result:
            this_screening_result = screening_result[key]
            result_df = pd.DataFrame({"sample_id": this_screening_result["sample_id"].view(-1).cpu().numpy(),
                                      "score": self.pred2score(this_screening_result[mol_prop_name])})
            pdb = key.split("@")[-1]
            result_df = result_df.astype({"sample_id": int}).set_index("sample_id")
            result_df = result_df.join(
                self.screening_reader.record_mapper[pdb].astype({"sample_id": int}).set_index("sample_id"))
            result_df["#code_ligand_num"] = result_df["ligand_file"].map(ligand_file2code)
            result_df = result_df[["#code_ligand_num", "score"]].sort_values(by=["#code_ligand_num"])
            result_df.to_csv(osp.join(self.save_root, screening_name, "pred_data", f"{pdb}_score.dat"), sep=" ",
                             index=False)

        result_summary = calc_screening_score(osp.join(self.save_root, screening_name), "pred_data", "screening")
        self.result_summary.update(result_summary)

    def screening_mixed_scores(self, n_mdn_lig, n_mdn_pose=None):
        """
        Use MDN probablity for docking decoy selections and pKd for binder selection.
        """
        if isinstance(n_mdn_lig, float):
            assert n_mdn_lig >=0. and n_mdn_lig <= 1.0, n_mdn_lig
            n_mdn_lig = math.ceil(285 * n_mdn_lig)
        screening_name = f"screening-mixed-lig_{n_mdn_lig}-pose_{n_mdn_pose}"
        os.makedirs(osp.join(self.save_root, screening_name, "pred_data"), exist_ok=True)

        # screening
        screening_result = self.screening_reader.result_mapper
        for key in tqdm(screening_result.keys(), desc=screening_name):
            this_screening_result = screening_result[key]
            info_df = pd.DataFrame({"sample_id": this_screening_result["sample_id"].view(-1).cpu().numpy(),
                                      "score": self.pred2score(this_screening_result["PROP_PRED"]),
                                      "score_mdn": this_screening_result["MDN_LOGSUM_DIST2_REFDIST2"].cpu().numpy()})
            pdb = key.split("@")[-1]
            info_df = info_df.astype({"sample_id": int}).set_index("sample_id")
            info_df = info_df.join(
                self.screening_reader.record_mapper[pdb].astype({"sample_id": int}).set_index("sample_id"))
            info_df["#code_ligand_num"] = info_df["ligand_file"].map(ligand_file2code)
            info_df = info_df.sort_values(by=["#code_ligand_num"])

            # Remove the predicted pKd of decoys. The decoys are determined by MDN probability
            info_df["binder_pdb"] = info_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
            binder_pdbs = set(info_df["binder_pdb"].values.tolist())
            result_df = []
            for binder_pdb in binder_pdbs:
                this_df = info_df[info_df["binder_pdb"] == binder_pdb].reset_index()
                if n_mdn_pose is not None:
                    # numpy argsort return accending order so I added "-"
                    mdn_decend = np.argsort(-this_df["score_mdn"].values)
                    this_df.loc[mdn_decend[n_mdn_pose:].tolist(), "score"] = -999999.
                result_df.append(this_df)

            if n_mdn_lig is not None:
                mdn_score_by_pdb = []
                for df in result_df:
                    this_pdb = df.loc[0, "binder_pdb"]
                    best_mdn_score = np.max(df["score_mdn"].values).item()
                    mdn_score_by_pdb.append((best_mdn_score, this_pdb))
                mdn_score_by_pdb.sort(key=lambda pair: pair[0], reverse=True)
                removed_pdbs = set([pair[1] for pair in mdn_score_by_pdb[n_mdn_lig:]])
                for df in result_df:
                    this_pdb = df.loc[0, "binder_pdb"]
                    if this_pdb in removed_pdbs:
                        df.loc[:, "score"] = -999999.

            result_df = pd.concat(result_df)
            result_df = result_df[["#code_ligand_num", "score"]]
            result_df.to_csv(osp.join(self.save_root, screening_name, "pred_data", f"{pdb}_score.dat"), sep=" ",
                             index=False)

        result_summary = calc_screening_score(osp.join(self.save_root, screening_name), "pred_data", "screening")
        self.result_summary.update(result_summary)

    def save_summary(self, save_name=None):
        df = pd.DataFrame(self.result_summary, index=[0])
        if save_name is None:
            save_name = "result_summary"
            if self.ref:
                save_name += "-ref"
        df.to_csv(osp.join(self.save_root, f"{save_name}.csv"), index=False)

    def docking_score(self, scoring_df, action: str="default"):
        # docking
        try:
            docking_result = self.docking_reader.result_mapper["test"]
        except FileNotFoundError:
            print("docking results not found, skipping...")
            return
        docking_name = "docking"
        if action == "default":
            mol_prop_name = "PROP_PRED"
            if self.use_mixed_scores:
                docking_name += "-pKd"
        elif action == "mdn":
            mol_prop_name = "PROP_PRED_MDN"
            docking_name += "-mdn"
        else:
            assert action.startswith("MDN_")
            mol_prop_name = action
            docking_name += "-"+action

        if self.ref:
            docking_name += "-ref"
        os.makedirs(osp.join(self.save_root, docking_name, "pred_data"), exist_ok=True)

        if "sample_id" in docking_result:
            sample_id = docking_result["sample_id"].view(-1).cpu().numpy()
        else:
            warnings.warn("sample_id not found, assuming the predicted property is in correct order.")
            sample_id = np.arange(len(docking_result[mol_prop_name]))
        result_dict = {"sample_id": sample_id, "score": self.pred2score(docking_result[mol_prop_name])}
        result_df = pd.DataFrame(result_dict)
        result_df = result_df.astype({"sample_id": int}).set_index("sample_id")
        record_keys = list(self.docking_reader.record_mapper.keys())
        assert len(record_keys) == 1
        result_df = result_df.join(
            self.docking_reader.record_mapper[record_keys[0]].astype({"sample_id": int}).set_index("sample_id"))
        result_df["#code"] = result_df["ligand_file"].map(ligand_file2code)
        try:
            result_df["pdb"] = result_df["protein_file"].map(_protein_file2pdb)
        except:
            result_df["pdb"] = result_df["#code"].map(lambda s: s.split("_")[0])
        pdb_set = set(result_df["pdb"].values.tolist())
        for pdb in pdb_set:
            this_pdb_df = result_df[result_df["pdb"] == pdb][["#code", "score"]].sort_values(by=["#code"])
            this_pdb_df = pd.concat([this_pdb_df, scoring_df.loc[[pdb]][["#code", "score"]]])
            this_pdb_df.to_csv(osp.join(self.save_root, docking_name, "pred_data", f"{pdb}_score.dat"),
                               sep=" ", index=False)
        
        # Deal with missing PDBs which is tipically cause by model not being able to parse specific PDBs
        tgt_df = pd.read_csv(osp.join(CASF_ROOT, "CASF-2016.csv"))
        required_pdbs = set(tgt_df["pdb"].values.tolist())
        missing_pdbs = required_pdbs.difference(pdb_set)
        assert len(missing_pdbs) <= 10, f"Too many missing PDBs: {len(missing_pdbs)}"
        for pdb in missing_pdbs:
            example_df = pd.read_csv(osp.join(CASF_ROOT, "power_docking", "examples", "X-Score", f"{pdb}_score.dat"), sep="\s+")
            example_df["score"] = [0.0] * example_df.shape[0]
            example_df.to_csv(osp.join(self.save_root, docking_name, "pred_data", f"{pdb}_score.dat"),
                               sep=" ", index=False)

        result_summary = calc_docking_score(osp.join(self.save_root, docking_name), "pred_data", "docking")
        self.result_summary.update(result_summary)

    def scoring_ranking_score(self, action: str="default"):
        if action == "default":
            mol_prop_name = "PROP_PRED"
            scatter_save_name = "exp_vs_cal.png"
        elif action == "mdn":
            mol_prop_name = "PROP_PRED_MDN"
            scatter_save_name = "exp_vs_cal_mdn.png"
        else:
            assert action.startswith("MDN_"), action
            mol_prop_name = action
            scatter_save_name = f"exp_vs_cal_{action}.png"

        # scoring
        scoring_result = self.scoring_reader.result_mapper["test"]
        record = self.scoring_reader.only_record()
        if "sample_id" in scoring_result:
            sample_id = scoring_result["sample_id"].view(-1).cpu().numpy()
        else:
            warnings.warn("sample_id not found, assuming the predicted property is in correct order.")
            sample_id = np.arange(len(scoring_result[mol_prop_name]))
        scoring_df = pd.DataFrame({"sample_id": sample_id, "score": self.pred2score(scoring_result[mol_prop_name])}).set_index("sample_id")
        scoring_df = scoring_df.join(record.astype({"sample_id": int}).set_index("sample_id"))
        scoring_df["#code"] = scoring_df["ligand_file"].map(lambda s: s.split(".sdf")[0].strip("[]'"))
        if len(scoring_df.iloc[0]["#code"]) != 4:
            scoring_df["#code"] = scoring_df["ligand_file"].map(lambda s: osp.basename(s).split("_")[0])
        if "protein_file" in scoring_df.columns:
            scoring_df["pdb"] = scoring_df["protein_file"].map(_protein_file2pdb)
        else:
            scoring_df["pdb"] = scoring_df["#code"].map(lambda s: s.split("_")[0])
        scoring_df = scoring_df.set_index("pdb")

        tgt_df = pd.read_csv(osp.join(CASF_ROOT, "CASF-2016.csv"))[["pdb", "pKd"]].set_index("pdb")
        scoring_df = scoring_df.join(tgt_df)

        exp = scoring_df["pKd"].values
        cal = scoring_df["score"].values
        r2 = plot_scatter_info(exp, cal, self.save_root, scatter_save_name, "Experimental pKd vs. Calculated pKd")
        # predicted pKd vs. MDN
        if "PROP_PRED_MDN" in scoring_result:
            pred_pkd = self.pred2score(scoring_result["PROP_PRED"])
            pred_mdn = self.pred2score(scoring_result["PROP_PRED_MDN"])
            try:
                plot_scatter_info(pred_pkd, pred_mdn, self.save_root, 
                                  "pkd_vs_mdn.png", "Predicted pKd vs. Predicted MDN", 
                                  xlabel="Predicted pKd", ylabel="Predicted MDN")
            except np.linalg.LinAlgError as e:
                print(e)

        # deal with missing pdbs
        required_pdbs = set(tgt_df.index.values.tolist())
        pdb_set = set(scoring_df.index.values.tolist())
        missing_pdbs = required_pdbs.difference(pdb_set)
        rank_info: pd.DataFrame = pd.read_csv(RANKING_CSV)
        pdb: List[str] = scoring_df.index.values.tolist()
        rank_pred: List[float] = cal.tolist()
        rank_exp: List[float] = exp.tolist()
        for missing_pdb in missing_pdbs:
            pdb.append(missing_pdb)
            rank_pred.append(-100.)
            rank_exp.append(tgt_df.loc[missing_pdb, "pKd"])
        rank_df = pd.DataFrame({"pdb": pdb, "PRED": rank_pred, "TGT": rank_exp}).set_index("pdb")
        rank_df = pd.concat([rank_df, rank_info.set_index("pdb")], axis=1)
        rank_df = rank_df.reset_index()
        spearman, kendall, details = get_rank(rank_df, rank_info)
        details["spearman_df"].to_csv(osp.join(self.save_root, f"spearman_df_{action}.csv"))
        details["kendall_df"].to_csv(osp.join(self.save_root, f"kendall_df_{action}.csv"))

        self.result_summary["score_r"] = math.sqrt(r2)
        self.result_summary["rank_spearman"] = spearman
        self.result_summary["rank_kendall"] = kendall
        return scoring_df

    def pred2score(self, prop_pred):
        if isinstance(prop_pred, torch.Tensor):
            prop_pred = prop_pred.cpu().numpy()
        else:
            assert isinstance(prop_pred, np.ndarray)

        if self.cfg["auto_pl_water_ref"]:
            return prop_pred[:, -1].reshape(-1) / pKd2deltaG
        else:
            return prop_pred.reshape(-1)

    @property
    def use_mixed_scores(self):
        if self._use_mixed_scores is None:
            # model are trained on both MDN loss and regression loss of pKd
            # as a result, model can predict both geometry probability as well as pKd
            self._use_mixed_scores = (self.cfg["loss_metric"].startswith("mdn_"))
        return self._use_mixed_scores

    @lazy_property
    def docking_reader(self):
        return TestedFolderReader(osp.basename(self.folder_name),
                    osp.basename(self.docking_test_folder),
                    osp.dirname(self.folder_name))

    @lazy_property
    def screening_reader(self):
        return TestedFolderReader(osp.basename(self.folder_name),
                    osp.basename(self.screening_test_folder),
                    osp.dirname(self.folder_name))

    @lazy_property
    def scoring_reader(self):
        return TestedFolderReader(osp.basename(self.folder_name),
                    osp.basename(self.scoring_test_folder),
                    osp.dirname(self.folder_name))

    @lazy_property
    def docking_test_folder(self):
        name = "casf2016-docking"
        if self.ref: name += "-ref"
        return self.get_folder(name)

    @lazy_property
    def scoring_test_folder(self):
        return self.get_folder("casf2016-scoring")

    @lazy_property
    def screening_test_folder(self):
        name = "casf2016-screening"
        if self.ref: name += "-ref"
        return self.get_folder(name)

    def get_folder(self, dataset):
        prefix = osp.basename(self.cfg["folder_prefix"])
        folders = glob(osp.join(self.folder_name, f"{prefix}_test_on_{dataset}_*"))
        folders.sort()
        assert len(folders) > 0
        return folders[-1]


def ligand_file2code(ligand_file: str):
    """
    Only used in CASF-2016 Docking score calculation
    """
    # remove the brackets which is unintentionally introduced during data processing
    if ligand_file.startswith("['"):
        ligand_file = ligand_file.split("['")[-1].split("']")[0]
    ligand_file = osp.basename(ligand_file).split(".pdb")[0]
    
    # if it is proprocessed to remove non-polar hydrogens, it looks like: 4ivc_1a30_single12 -> 1a30_ligand_12
    if "_single" in ligand_file:
        ligand_file = "_".join(ligand_file.split("_")[1:])
        ligand_file = ligand_file.replace("_single", "_ligand_")
    return ligand_file


def _protein_file2pdb(protein_file):
    if protein_file.startswith("['"):
        protein_file = protein_file.split("['")[-1].split("']")[0]
    return osp.basename(protein_file).split("_")[0]


def plot_scatter_info(exp, cal, save_folder, save_name, title, total_time=None, original_size=None, 
                      xlabel="Experimental", ylabel="Calculated", return_r=False):
    """
    Scatter plots and scoring power
    :param exp:
    :param cal:
    :param save_folder:
    :param save_name:
    :param title:
    :param total_time:
    :param original_size:
    :return:
    """
    mae = np.mean(np.abs(exp - cal))
    rmse = np.sqrt(np.mean((exp - cal) ** 2))
    r = pearsonr(exp, cal)[0]
    r2 = r ** 2

    plt.figure()
    xy = np.vstack([exp, cal])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    plt.scatter(exp[idx], cal[idx], c=z[idx])
    x_min = min(exp)
    x_max = max(exp)
    plt.plot([x_min, x_max], [x_min, x_max], color="black", label="y==x", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    annotate = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nPearson R = {r:.4f}\n"
    if total_time is not None:
        annotate = annotate + f"Total Time = {total_time:.0f} seconds\n"
    if original_size is not None:
        annotate = annotate + f"Showing {len(exp)} / {original_size}\n"
    plt.annotate(annotate, xy=(0.05, 0.65), xycoords='axes fraction')

    # Linear regression
    lr_model = LinearRegression()
    lr_model.fit(exp.reshape(-1, 1), cal)
    slope = lr_model.coef_.item()
    intercept = lr_model.intercept_.item()
    plt.plot([x_min, x_max], [lr_model.predict(x_min.reshape(-1, 1)), lr_model.predict(x_max.reshape(-1, 1))],
             label="y=={:.2f}x+{:.2f}".format(slope, intercept), color="red")

    plt.legend()
    plt.savefig(osp.join(save_folder, save_name))
    plt.close()
    if return_r:
        return r
    return r2


def get_rank(pred: pd.DataFrame, rank_info: pd.DataFrame):
    """
    Ranking calculation, adapted from
    https://github.com/cyangNYU/delta_LinF9_XGB/blob/main/performance/Train_validation_test_performances.ipynb
    """
    spearman_list, kendall_list, target, pdb = [], [], [], []
    for i in range(1, 58):
        sa = rank_info.loc[rank_info['target'] == i]
        target.append(i)
        pdb.append(sa.pdb.tolist())
        de = pred.loc[pred['pdb'].isin(sa.pdb.tolist())]
        spearman_list.append(round(spearmanr(de['PRED'], de['pKd'])[0], 3))
        kendall_list.append(round(kendalltau(de['PRED'], de['pKd'])[0], 3))
    spearman_df = pd.DataFrame({"spearman": spearman_list, "target#": np.arange(1, 58)}).set_index("target#")
    kendall_df = pd.DataFrame({"kendall": kendall_list, "target#": np.arange(1, 58)}).set_index("target#")
    details = {"spearman_df": spearman_df, "kendall_df": kendall_df}
    return np.mean(spearman_list), np.mean(kendall_list), details

# base environment python with pandas==1.5.0
# the CASF-2016 test script does not work with newer version of pandas.
PY="/ext3/miniconda3/envs/old-pandas/bin/python"

def calc_docking_score(run_dir, score_dir, out_name):
    script = osp.join(CASF_ROOT, "power_docking", "docking_power.py")
    core_set = osp.join(CASF_ROOT, "power_docking", "CoreSet.dat")
    result = osp.join(run_dir, score_dir)
    decoy = osp.join(CASF_ROOT, "decoys_docking")
    out = osp.join(run_dir, f"model_{out_name}")
    out_print = osp.join(run_dir, f"{out_name}.out")
    subprocess.run(
        f"{PY} {script} -c {core_set} -s {result} -r {decoy} -p 'positive' -l 2 -o '{out}' > {out_print}",
        shell=True, check=True)

    result_summary = {}
    with open(out_print) as f:
        result_lines = f.readlines()
        for i, line in enumerate(result_lines):
            if line.startswith("Among the top1 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR1"] = result_lines[i+1].split()[-1]
            elif line.startswith("Among the top2 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR2"] = result_lines[i+1].split()[-1]
            elif line.startswith("Among the top3 binding pose ranked by the given scoring function:"):
                result_summary["docking_SR3"] = result_lines[i+1].split()[-1]
    return result_summary

def calc_screening_score(run_dir: str, score_dir: str, out_name: str):
    # A python wrapper to call the CASF-2016 screening power
    # run_dir: all output files are saved here
    # score_dir: raw score csv are saved here
    # out_name: output prefix
    # forward screening power
    script = osp.join(CASF_ROOT, "power_screening", "forward_screening_power.py")
    core_set = osp.join(CASF_ROOT, "power_screening", "CoreSet.dat")
    result = osp.join(run_dir, score_dir)
    target_info = osp.join(CASF_ROOT, "power_screening", "TargetInfo.dat")
    out = osp.join(run_dir, f"model_{out_name}")
    out_print = osp.join(run_dir, f"{out_name}.out")
    subprocess.run(f"{PY} {script} -c {core_set} -s {result} -t {target_info} -p 'positive' -o {out} > {out_print}",
                    shell=True, check=True)

    result_summary = {}
    with open(out_print) as f:
        result_lines = f.readlines()
        for line in result_lines:
            if line.startswith("Average enrichment factor among top 1% = "):
                result_summary["screening_EF1"] = line.split()[-1]
            elif line.startswith("Average enrichment factor among top 5% = "):
                result_summary["screening_EF5"] = line.split()[-1]
            elif line.startswith("Average enrichment factor among top 10% = "):
                result_summary["screening_EF10"] = line.split()[-1]
            elif line.startswith("The best ligand is found among top 1% candidates"):
                result_summary["screening_SR1"] = line.split()[-1]
            elif line.startswith("The best ligand is found among top 5% candidates"):
                result_summary["screening_SR5"] = line.split()[-1]
            elif line.startswith("The best ligand is found among top 10% candidates"):
                result_summary["screening_SR10"] = line.split()[-1]

    # reverse screening power
    script_rev = osp.join(CASF_ROOT, "power_screening", "reverse_screening_power.py")
    ligand_info = osp.join(CASF_ROOT, "power_screening", "LigandInfo.dat")
    out = osp.join(run_dir, f"model_{out_name}_rev")
    out_print = osp.join(run_dir, f"{out_name}_rev.out")
    subprocess.run(f"{PY} {script_rev} -c {core_set} -s {result} -l {ligand_info} -p 'positive' -o {out} > {out_print}",
                    shell=True, check=True)

    with open(out_print) as f:
        result_lines = f.readlines()
        for line in result_lines:
            if line.startswith("The best target is found among top 1% candidates"):
                result_summary["screening_rev_SR1"] = line.split()[-1]
            elif line.startswith("The best target is found among top 5% candidates"):
                result_summary["screening_rev_SR5"] = line.split()[-1]
            elif line.startswith("The best target is found among top 10% candidates"):
                result_summary["screening_rev_SR10"] = line.split()[-1]
    return result_summary


if __name__ == '__main__':
    calculator = CasfScoreCalculator("../exp_pl_005_run_2022-06-03_164656__676997")
    calculator.run()
