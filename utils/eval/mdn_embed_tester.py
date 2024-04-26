import pickle
import os
import os.path as osp
import numpy as np
import pandas as pd

from utils.scores.casf_scores import calc_docking_score, get_rank, ligand_file2code, plot_scatter_info
from utils.eval.tester import Tester
from utils.train.mdn2pkd_trainer import CPU_Unpickler

class MDNEmbedTester(Tester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.result_info = {}
        self.casf_folder = osp.join(self.folder_name, "casf-scores")
        os.makedirs(self.casf_folder, exist_ok=True)

        self._scaler = None
        self._ds_root = None

    def run_test(self):
        scoring_df = self.run_casf_score_rank()
        self.run_casf_dock(scoring_df)

        res_df = pd.DataFrame(self.result_info, index=[0])
        res_df.to_csv(osp.join(self.casf_folder, "result_summary.csv"), index=False)

    def run_casf_score_rank(self):
        y_pred, y_labels, ds = self.test_on_ds("casf-scoring.polar.polar.pl358.mdnembed.pickle")

        r_casf = plot_scatter_info(y_labels, y_pred, self.casf_folder, "exp_vs_cal", 
                    "Experimental pKd vs. Calculated pKd", return_r=True)
        score_mae = np.mean(np.abs(y_labels - y_pred))

        pred_info = {"pdb": ds["pdb"], "pKd": y_labels, "PRED": y_pred}
        pred_df = pd.DataFrame(pred_info)
        rank_df = pd.read_csv("/vast/sx801/geometries/CASF-2016-cyang/CASF-2016.csv")
        spearman, kendall, details = get_rank(pred_df, rank_df)

        res = {"score_r": r_casf, "score_mae": score_mae,
               "rank_spearman": spearman, "rank_kendall": kendall}
        self.result_info.update(res)

        scoring_df = pred_df[["pdb", "pKd"]].rename({"pKd": "score"}, axis=1)
        scoring_df["#code"] = scoring_df["pdb"]
        scoring_df = scoring_df.set_index("pdb")
        return scoring_df

    def run_casf_dock(self, scoring_df: pd.DataFrame):
        y_pred, y_labels, ds = self.test_on_ds("casf-docking.polar.polar.pl358.mdnembed.pickle")
        dock_df: pd.DataFrame = pd.DataFrame({"score": y_pred, "pdb": ds["pdb"], 
                "#code": [ligand_file2code(l[0]) for l in ds["ligand_file"]]})
        pdb_set = set(dock_df["pdb"].values.tolist())
        os.makedirs(osp.join(self.casf_folder, "docking", "pred_data"), exist_ok=True)
        for pdb in pdb_set:
            this_pdb_df = dock_df[dock_df["pdb"] == pdb][["#code", "score"]].sort_values(by=["#code"])
            this_pdb_df = pd.concat([this_pdb_df, scoring_df.loc[[pdb]][["#code", "score"]]])
            this_pdb_df.to_csv(osp.join(self.casf_folder, "docking", "pred_data", f"{pdb}_score.dat"),
                               sep=" ", index=False)

        result_summary = calc_docking_score(osp.join(self.casf_folder, "docking"), "pred_data", "docking")
        self.result_info.update(result_summary)

    def test_on_ds(self, ds_name: str):
        with open(osp.join(self.ds_root, ds_name), "rb") as f:
            ds = CPU_Unpickler(f).load()
        x_feats = ds[f"{self.cfg['mdn_embed_type']}_embed"]
        x_feats = np.concatenate([t.numpy().reshape(1, -1) for t in x_feats], axis=0)
        x_feats_scaled = self.scaler.transform(x_feats)
        y_pred = self.model.predict(x_feats_scaled)
        if "pKd" in ds:
            y_labels = ds["pKd"]
            y_labels = np.asarray([i.item() for i in y_labels])
        else:
            y_labels = None
        return y_pred, y_labels, ds

    @property
    def model(self):
        if self._model is not None:
            return self._model
        with open(osp.join(self.folder_name, "model.pkl"), "rb") as f:
            self._model = pickle.load(f)
        return self._model

    @property
    def scaler(self):
        if self._scaler is not None:
            return self._scaler
        with open(osp.join(self.folder_name, "scaler.pkl"), "rb") as f:
            self._scaler = pickle.load(f)
        return self._scaler

    @property
    def ds_root(self):
        if self._ds_root is not None:
            return self._ds_root
        ds_args = self.cfg if self.explicit_ds_args is None else self.explicit_ds_args
        self._ds_root = osp.join(ds_args["data_root"], "processed")
        return self._ds_root
