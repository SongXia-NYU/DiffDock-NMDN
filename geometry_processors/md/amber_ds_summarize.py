import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os.path as osp
import os
import json
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import logging
from prody import parsePDB

from geometry_processors.pl_dataset.all2single_pygs import ProteinProcessor
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.misc import solv_num_workers


def mp_process_prot(args):
    pdb_f, gbsa_folder, term, pyg_root, p_min, p_max = args
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)
    prop_dict = {}
    af_id = osp.basename(pdb_f).split(".amber.H.martini.pdb")[0]

    try:
        energy = {}
        for phase in ["gas", "water", "octanol"]:
            this_e = pd.read_csv(osp.join(gbsa_folder, f"csvs.{phase}", f"{af_id}.{phase}.csv"))[term].item()
            energy[phase] = float(this_e)
            if energy[phase] < p_min or energy[phase] > p_max:
                raise ValueError(f"Property not within range [{p_min}, {p_max}]: {energy[phase]}")
        prop_dict["gasEnergy"] = energy["gas"]
        prop_dict["watEnergy"] = energy["water"]
        prop_dict["octEnergy"] = energy["octanol"]
        prop_dict["CalcSol"] = energy["water"] - energy["gas"]
        prop_dict["CalcOct"] = energy["octanol"] - energy["gas"]
        prop_dict["watOct"] = energy["water"] - energy["octanol"]

        mp_info = MPInfo(protein_pdb=pdb_f, cutoff_protein=10., pyg_name=f"{pyg_root}/{af_id}.pth", **prop_dict)

        processor = ProteinProcessor(mp_info, martini=True)

        processor.process_single_entry()
    except Exception as e:
        out = {"af_id": af_id, "Error": str(e)}
        return out


class TriSolvSummarizer:
    def __init__(self, root, save_root, seq_stat=False, e_cap=np.inf) -> None:
        self.root = root
        self.save_root = save_root
        self.seq_stat = seq_stat
        self.e_cap = e_cap

        self._csvs = None
        self._summary_dfs = None
        self._vis_root = None

    def prepare_ds(self, martini_folder, gbsa_folder, term, pyg_root, p_min, p_max, frag_size):
        mp_args = [(f, gbsa_folder, term, pyg_root, p_min, p_max) for f in
                   glob(osp.join(martini_folder, "pdb", "*.pdb"))]
        n_cpu_avail, n_cpu, num_workers = solv_num_workers()
        errors = process_map(mp_process_prot, mp_args, chunksize=20, max_workers=num_workers)
        errors = [e for e in errors if e is not None]
        return errors

    def run(self):
        json_sum = {}
        os.makedirs(self.save_root, exist_ok=True)
        vis_terms = ["ENERGY", "BOND", "ANGLE", "DIHED", "VDWAALS", "EEL", "EGB", "1-4 VDW", "1-4 EEL", "RESTRAINT",
                     "ESURF"]
        trans_df = []

        for phase in self.summary_dfs:
            this_mask = self.summary_dfs[phase][["ENERGY"]].applymap(np.isreal).values
            this_mask = this_mask & self.summary_dfs[phase][["ENERGY"]].applymap(lambda x: x < self.e_cap).values
            this_mask = this_mask & self.summary_dfs[phase][["VDWAALS"]].applymap(lambda x: x != "*************").values

            this_df = self.summary_dfs[phase].loc[this_mask.reshape(-1), :].astype({"VDWAALS": float})
            json_sum[f"{phase[1:]}_total_time"] = this_df["Total time"].sum()

            trans_df.append(this_df.add_prefix(f"{phase[1:]}_"))

            vis_folder = osp.join(self.vis_root, phase[1:])
            os.makedirs(vis_folder, exist_ok=True)
            for term in vis_terms:
                plt.figure(figsize=(8, 6))
                ax = sns.histplot(this_df, x=term, bins=25, log_scale=False)
                ax.set(xlabel=f"{phase[1:].upper()} {term}")
                plt.title(f"N_total={this_df.shape[0]}")
                plt.tight_layout()
                plt.savefig(osp.join(vis_folder, f"dist_{term.replace(' ', '_')}.png"))
                plt.close()

        trans_df = pd.concat(trans_df, axis=1)
        vis_folder = osp.join(self.vis_root, f"transfer")
        os.makedirs(vis_folder, exist_ok=True)
        for p1, p2 in (("water", "gas"), ("octanol", "gas"), ("water", "octanol")):
            for term in vis_terms:
                trans_df[f"{p1}_{p2}_{term}"] = trans_df[f"{p1}_{term}"] - trans_df[f"{p2}_{term}"]
                plt.figure(figsize=(8, 6))
                sns.histplot(trans_df, x=f"{p1}_{p2}_{term}", bins=25, log_scale=False)
                plt.title(f"N_total={trans_df.shape[0]}")
                plt.tight_layout()
                plt.savefig(osp.join(vis_folder, f"{term}_{p1}_{p2}.png"))
                plt.close()
        trans_df.to_csv(osp.join(self.save_root, f"summary.transfer.csv"))

        trans_df = trans_df.sort_values(by="gas_ENERGY", ascending=False)
        top_10_idx = trans_df.iloc[:10].index.values.tolist()
        json_sum["TOP10_GAS_ENERGY"] = top_10_idx

        with open(osp.join(self.save_root, "summary.json"), "w") as f:
            json.dump(json_sum, f, indent=2)

        if self.seq_stat:
            vis_folder = osp.join(self.vis_root, "seq_stat")
            os.makedirs(vis_folder, exist_ok=True)
            all_seqs = set(trans_df["gas_seq"].values.tolist())
            for seq in all_seqs:
                this_df = trans_df[trans_df["gas_seq"] == seq]
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
                for i, prop in enumerate(
                        ["gas_ENERGY", "water_ENERGY", "octanol_ENERGY", "water_gas_ENERGY", "octanol_gas_ENERGY",
                         "water_octanol_ENERGY"]):
                    sns.histplot(this_df, x=prop, ax=axes[i])
                plt.suptitle(f"N_total={this_df.shape[0]}")
                plt.tight_layout()
                plt.savefig(osp.join(vis_folder, f"{seq}.png"))
                plt.close()

    @property
    def vis_root(self):
        if self._vis_root is None:
            if self.e_cap == np.inf:
                self._vis_root = self.save_root
            else:
                self._vis_root = osp.join(self.save_root, f"ecap_{self.e_cap}")
        return self._vis_root

    @property
    def summary_dfs(self):
        if self._summary_dfs is None:
            out = {}
            tik = 0
            for phase in [".gas", ".water", ".octanol"]:
                if osp.exists(osp.join(self.save_root, f"summary{phase}.csv")):
                    out[phase] = pd.read_csv(osp.join(self.save_root, f"summary{phase}.csv"), index_col=0)
                    tik += 1
            if tik == 3:
                self._summary_dfs = out
                return self._summary_dfs

            for phase in self.csvs:
                if phase in out.keys():
                    continue

                out[phase] = pd.concat([self.read_csv(f) for f in tqdm(self.csvs[phase])], axis=0)
                out[phase].to_csv(osp.join(self.save_root, f"summary{phase}.csv"))
            self._summary_dfs = out
        return self._summary_dfs

    def read_csv(self, csv):
        df = pd.read_csv(csv, index_col=0)
        if self.seq_stat:
            pdb_f = osp.join(self.root, "amber_h_pdbs", df.index.item() + ".amber.H.pdb")
            if osp.exists(pdb_f):
                atoms = parsePDB(pdb_f)
                seq_out = ""
                seq = atoms.getSequence()
                res_id = atoms.getResindices()
                prev_res = -1
                for res_i, s in zip(res_id, seq):
                    if res_i != prev_res:
                        seq_out += s
                        prev_res = res_i
            else:
                seq_out = "UNK"
            df["seq"] = [seq_out]
        return df

    @property
    def csvs(self):
        if self._csvs is None:
            out = {}
            for phase in [".gas", ".water", ".octanol"]:
                out[phase] = glob(osp.join(self.root, f"csvs{phase}", "*"))
            self._csvs = out
        return self._csvs


if __name__ == "__main__":
    viser = TriSolvSummarizer("/vast/sx801/AF-SwissProt500-GBSA",
                              "/scratch/sx801/scripts/Mol3DGenerator/data/AF-SwissProt/AF-SwissProt500-GBSA")
    viser.run()
