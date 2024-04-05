from collections import defaultdict
from functools import cached_property
from typing import List, Optional, Set
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob

import pandas as pd
from urllib.request import urlretrieve


class NonbinderReader:
    def __init__(self, ds_root: Optional[str] = None) -> None:
        if ds_root is None:
            ds_root = "/vast/sx801/geometries/Yaowen_nonbinders"
        self.ds_root = ds_root

    def uniprot_id2prot_polarh(self, uniprot_id: str) -> str:
        return osp.join(self.ds_root, "protein_pdbs_polarh", f"{uniprot_id}.pdb")

    def uniprot_id2prot_og(self, uniprot_id: str) -> str:
        return osp.join(self.ds_root, "protein_pdbs", f"{uniprot_id}.pdb")

    def fl2ligs(self, fl: str) -> List[str]:
        return glob(osp.join(self.ds_root, "pose_diffdock", "raw_predicts", fl, "rank*_confidence*.sdf"))

    @cached_property
    def info_df(self) -> pd.DataFrame:
        INTERESED_COLS = ["Activity", "SMILES", "Uniprot_ID"]
        ic50_csv = osp.join(self.ds_root, "ic50.csv")
        ic50_df = pd.read_csv(ic50_csv)
        ic50_df = ic50_df[ic50_df["Activity"] <= 4.0][INTERESED_COLS]
        ic50_df["file_handle"] = ic50_df.index.map(lambda i: f"nonbinders.ic50.{i}")

        ki_csv = osp.join(self.ds_root, "ki.csv")
        ki_df = pd.read_csv(ki_csv)
        ki_df = ki_df[ki_df["Activity"] <= 4.0][INTERESED_COLS]
        ki_df["file_handle"] = ki_df.index.map(lambda i: f"nonbinders.ki.{i}")
        info_df = pd.concat([ic50_df, ki_df], axis=0)
        return info_df

    @cached_property
    def uniprot_ids(self) -> Set[str]:
        unique_ids = set(self.info_df["Uniprot_ID"].values.reshape(-1).tolist())
        return unique_ids

    def download_prot_pdbs(self) -> None:
        savedir = osp.join(self.ds_root, "protein_pdbs")
        os.makedirs(savedir, exist_ok=True)
        for uniprot_id in tqdm(self.uniprot_ids):
            urlretrieve(f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb", 
                        osp.join(savedir, f"{uniprot_id}.pdb"))

    def generate_diffdock_jobs(self) -> None:
        save_root = "/scratch/sx801/temp/DiffDock_nonbinders_jobs"
        n_gpus = 40
        all_job_info = defaultdict(lambda: [])
        for i in range(self.info_df.shape[0]):
            this_info = self.info_df.iloc[i]
            all_job_info["complex_name"].append(this_info["file_handle"])
            uniprot_id = this_info["Uniprot_ID"]
            prot_path = osp.join(self.ds_root, "protein_pdbs", f"{uniprot_id}.pdb")
            all_job_info["protein_path"].append(prot_path)
            all_job_info["ligand_description"].append(this_info["SMILES"])
            all_job_info["protein_sequence"].append("")
        all_job_df = pd.DataFrame(all_job_info).sample(frac=1.)
        for i, df_chunk in enumerate(np.array_split(all_job_df, n_gpus)):
            df_chunk.to_csv(osp.join(save_root, f"job_gpu_{i}.csv"), index=False)

if __name__ == "__main__":
    print(NonbinderReader().info_df.set_index("file_handle").loc["nonbinders.ic50.1000415"])
