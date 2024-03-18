from collections import defaultdict
import json
import logging
import math
import os
import os.path as osp
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

from utils.eval.TestedFolderReader import TestedFolderReader
from utils.scores.casf_scores import plot_scatter_info
from utils.eval.trained_folder import TrainedFolder


class ProtPepScoreCalculator(TrainedFolder):
    def __init__(self, folder_name):
        super().__init__(folder_name)

        testdir = glob(osp.join(folder_name, "exp_ppep_???_test_on_prot_pep_test_af2_*"))[0]
        self.test_reader = TestedFolderReader(folder_name, osp.basename(testdir), root=None)
        self.DS_ROOT = "/vast/sx801/geometries/ProtPep_Xuhang"
        self.dockq_csv = osp.join(self.DS_ROOT, "label_n_split", "song_biolip_2369_full.csv")
        self._save_root = osp.join(folder_name, "prot_pep_scores")
        os.makedirs(self._save_root, exist_ok=True)

        self._dockq_df = None
        self._exp_rank_id = None
        self._exp_rank_dockq = None
        self._model_rank_id = None
        self._model_pred_df = None
        self._dockq_query = None
        self._dockq_query_df = None

    @property
    def save_root(self):
        return self._save_root

    @property
    def dockq_df(self):
        if self._dockq_df is not None:
            return self._dockq_df
        
        self._dockq_df = pd.read_csv(self.dockq_csv)
        return self._dockq_df
    
    @property
    def dockq_query(self):
        if self._dockq_query is not None:
            return self._dockq_query
        
        dockq_query = {}
        for i in range(self.dockq_df.shape[0]):
            this_df = self.dockq_df.iloc[i]
            pdb_id = this_df["pdb_id"]
            model_id = this_df["af_model_id"]
            dockq = this_df["pdb2sql_DockQ"]
            dockq_query[(pdb_id, model_id)] = dockq

        dockq_query_df = defaultdict(lambda: [])
        for key in dockq_query.keys():
            dockq_query_df["pdb"].append(key[0])
            dockq_query_df["model_id"].append(key[1])
            dockq_query_df["dockq"].append(dockq_query[key])
        dockq_query_df = pd.DataFrame(dockq_query_df).set_index(["pdb", "model_id"])
        self._dockq_query_df = dockq_query_df
        self._dockq_query = dockq_query
        return self._dockq_query
    
    @property
    def dockq_query_df(self):
        if self._dockq_query_df is None:
            __ = self.dockq_query
        return self._dockq_query_df
    
    @property
    def exp_rank_id(self) -> dict:
        # pdb_id -> the af_model_ids of the structure with the highest dockQ
        if self._exp_rank_id is not None:
            return self._exp_rank_id
        
        top_rank_id = {}
        top_rank_dockq = {}
        test_df = self.dockq_df[self.dockq_df["data_class"] == "test"]
        # select the rows with the largest dockq for each pdb_id
        test_df = test_df.sort_values("pdb2sql_DockQ", ascending=False).drop_duplicates(["pdb_id"]).set_index("pdb_id")
        all_pdbs = set(test_df.index.tolist())
        for pdb in all_pdbs:
            top_rank_id[pdb] = test_df.loc[pdb, "af_model_id"]
            top_rank_dockq[pdb] = test_df.loc[pdb, "pdb2sql_DockQ"]
        self._exp_rank_id = top_rank_id
        self._exp_rank_dockq = top_rank_dockq
        return self._exp_rank_id
    
    @property
    def exp_rank_dockq(self) -> dict:
        # pdb_id -> the highest dockq
        if self._exp_rank_dockq is not None:
            return self._exp_rank_dockq
        
        __ = self.exp_rank_dockq
        return self._exp_rank_dockq
    
    @property
    def model_rank_id(self) -> dict:
        if self._model_rank_id is not None:
            return self._model_rank_id
        
        record_df: pd.DataFrame = self.test_reader.only_record().set_index("sample_id")
        pred_info = {"sample_id": self.test_reader.result_mapper["test"]["sample_id"].cpu(),
                     "pred_score": self.test_reader.result_mapper["test"]["PROP_PRED"].cpu().view(-1)}
        pred_df = pd.DataFrame(pred_info).set_index("sample_id")
        record_df = record_df.join(pred_df)
        def prot_file2model_id(prot_file: str):
            return int(osp.basename(prot_file).split("_")[1])
        record_df["model_id"] = record_df["protein_file"].map(prot_file2model_id)

        all_pdbs = set(record_df["pdb"].values.tolist())
        top_rank_model = {}
        for pdb in all_pdbs:
            this_df = record_df[record_df["pdb"] == pdb].sort_values(by="pred_score", ascending=False)
            model_id = this_df.iloc[0]["model_id"]
            top_rank_model[pdb] = model_id
        self._model_rank_id = top_rank_model
        self._model_pred_df = record_df[["pred_score", "pdb", "model_id"]].set_index(["pdb", "model_id"])
        return self._model_rank_id
    
    @property
    def model_pred_df(self):
        if self._model_pred_df is None:
            __ = self._model_pred_df
        return self._model_pred_df

    def run(self):
        # success rate of AF2
        self.logger.setLevel(logging.INFO)
        self.logger.info("hello there")

        # the model_id with the highest dockq
        exp_ids = []
        # the model_id selected by the model
        model_sel_ids = []
        # the highest dockq
        exp_dockq = []
        # the dockq selected by the model
        model_sel_dockq = []

        #------ Model selected top1 DockQ vs. Highest possible DockQ ------#
        for pdb_id in self.exp_rank_id:
            exp_ids.append(self.exp_rank_id[pdb_id])
            model_sel_id = self.model_rank_id[pdb_id]
            model_sel_ids.append(model_sel_id)
            exp_dockq.append(self.exp_rank_dockq[pdb_id])
            key = (pdb_id, model_sel_id)
            # sometimes the query does not contain the key, assume it to be 0.
            if key not in self.dockq_query:
                model_sel_dockq.append(0.)
                logging.warn(f"{key} not in self.query. Assuming DockQ to be 0.!!")
                continue
            model_sel_dockq.append(self.dockq_query[key])
        
        save_img = osp.join(self.save_root, "model_success_rate.png")
        res_summary = draw_selction_sr(exp_ids, model_sel_ids, exp_dockq, model_sel_dockq, save_img, 
                         title="Model Selected Top1 DockQ vs. Highest DockQ")
        
        #------ Model predicted Score vs. Real DockQ ------#
        compare_df = self.model_pred_df.join(self.dockq_query_df).dropna()
        r2 =plot_scatter_info(compare_df["dockq"].values, compare_df["pred_score"].values, 
                            self.save_root, "pred_score_vs_docq.png",
                            "Model predicted Score vs. DockQ", xlabel="DockQ", ylabel="Predicted Score")
        r = math.sqrt(r2)
        res_summary["pearson_r"] = r
        with open(osp.join(self.save_root, "res_summary.json"), "w") as f:
            json.dump(res_summary, f, indent=2)


    def run_af2(self):
        # the model_id with the highest dockq
        exp_ids = []
        # the model_id selected by the model
        model_sel_ids = []
        # the highest dockq
        exp_dockq = []
        # the dockq selected by the model
        model_sel_dockq = []

        for pdb_id in self.exp_rank_id:
            exp_ids.append(self.exp_rank_id[pdb_id])
            model_sel_ids.append(0)
            exp_dockq.append(self.exp_rank_dockq[pdb_id])
            model_sel_dockq.append(self.dockq_query[(pdb_id, 0)])
        
        save_img = osp.join(self.save_root, "af2_success_rate.png")
        draw_selction_sr(exp_ids, model_sel_ids, exp_dockq, model_sel_dockq, save_img)

    
def draw_selction_sr(exp_ids, model_sel_ids, exp_dockq, model_sel_dockq, save_img, 
                     threshold=0.04, title=None):
    n_total = len(exp_ids)
    n_success = 0
    logging.info(f"Test set size: {n_total}")
    for sel_dockq, best_dockq in zip(model_sel_dockq, exp_dockq):
        if math.fabs(sel_dockq - best_dockq) <= threshold:
            n_success += 1
    logging.info(f"N Success: {n_success}")
    logging.info(f"Success rate: {1.0 * n_success / n_total}")

    n_success_strict = 0
    for exp_id, model_sel_id in zip(exp_ids, model_sel_ids):
        if exp_id == model_sel_id:
            n_success_strict += 1
    logging.info(f"N Success Strict: {n_success_strict}")
    strict_sr = 1.0 * n_success_strict / n_total
    logging.info(f"Strict Success rate: {strict_sr}")

    fig, ax = plt.subplots(1, 1)
    plt.plot([0., 1.], [0., 1.], label="Upper Bound: y==x", color="red")
    plt.plot([threshold, 1.], [0., 1.-threshold], label=f"Threshold: {threshold}", linestyle="--", color="black")
    success_rate = "{:.2f} %".format(100.0 * n_success / n_total)
    annotate = f"Success: {n_success} out of {n_total} \nSuccess rate: {success_rate}"
    plt.annotate(annotate, xy=(0.05, 0.65))

    sns.scatterplot(x=exp_dockq, y=model_sel_dockq, alpha=0.5)
    ax.set_xlabel("Highest DockQ")
    ax.set_ylabel("Selected Top1 DockQ")
    if title is None:
        title = "AF2 Selected Top1 DockQ vs. Highest DockQ"
    ax.set_title(title)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_img)
    plt.close()

    return {f"success_rate_{threshold}": success_rate, f"strict_success_rate_{threshold}": strict_sr,
            "n_total": n_total, "n_success": n_success, "n_success_strict": n_success_strict}
