from collections import Counter, defaultdict
from functools import cached_property
from typing import Dict, List, Optional, Set
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob

import pandas as pd
from urllib.request import urlretrieve


class NonbinderReader:
    def __init__(self, ds_root: Optional[str] = None, threshold=5.0) -> None:
        if ds_root is None:
            ds_root = "/vast/sx801/geometries/Yaowen_nonbinders"
        self.ds_root = ds_root
        self.threshold = threshold

    def uniprot_id2afprot_polarh(self, uniprot_id: str) -> str:
        return osp.join(self.ds_root, "protein_pdbs_polarh", f"{uniprot_id}.pdb")

    def uniprot_id2afprot_og(self, uniprot_id: str) -> str:
        return osp.join(self.ds_root, "protein_pdbs", f"{uniprot_id}.pdb")

    def pdb2prot_og(self, pdb: str) -> str:
        return osp.join(self.ds_root, "protein_crystal_pdbs", "structures", f"{pdb}.pdb")

    def pdb2prot_poarlh(self, pdb: str) -> str:
        return osp.join(self.ds_root, "protein_crystal_pdbs", "polarh", f"{pdb}.pdb")

    def fl2afligs(self, fl: str) -> List[str]:
        return glob(osp.join(self.ds_root, "pose_diffdock", "raw_predicts", fl, "rank*_confidence*.sdf"))

    def fl2crysligs(self, fl: str) -> List[str]:
        return glob(osp.join(self.ds_root, "pose_diffdock_crystal", "raw_predicts", fl, "rank*_confidence*.sdf"))

    @cached_property
    def sampled_pl_info_df(self) -> pd.DataFrame:
        # self.best_res_info_df contains over 300,000 entries, which is too large
        # here we only sample 60,000 of them.
        save_csv = osp.join(self.ds_root, "sampled_pl_info.csv")
        if osp.exists(save_csv):
            return pd.read_csv(save_csv, index_col="Uniprot_ID")
        all_uniprot_ids = self.best_res_info_df.index.values.reshape(-1).tolist()
        uniprot_id_counts = Counter(all_uniprot_ids)
        uniprot_id_weights = {key: 1.0/uniprot_id_counts[key] for key in uniprot_id_counts}
        weights = [uniprot_id_weights[key] for key in all_uniprot_ids]
        sampled = self.best_res_info_df.sample(n=60_000, weights=weights, random_state=2342)
        sampled.to_csv(save_csv)
        return sampled

    @cached_property
    def best_res_info_df(self) -> pd.DataFrame:
        # crystal structure retrieved from RCSB. For those uniprot-id with multiple PDB entries,
        # the highest resolution one is retrieved.
        save_csv = osp.join(self.ds_root, "best_res_info.filtered.csv")
        if osp.exists(save_csv):
            return pd.read_csv(save_csv, index_col="Uniprot_ID")
        # pandas dataframe recording the crystal sctructure with the crystal structure 
        # with the highest resolution
        pdb2res: Dict[str, float] = {}
        res_df = pd.read_csv(osp.join(self.ds_root, "protein_crystal_pdbs", "res_info.csv"))
        for i in range(res_df.shape[0]):
            this_info = res_df.iloc[i]
            pdb = this_info["pdb"]
            res = this_info["resolution"]
            pdb2res[pdb] = res
        
        id_mapping_df = pd.read_csv(osp.join(self.ds_root, "idmapping_2024_04_07.tsv"), sep="\t")
        id_mapping_df["resolution"] = id_mapping_df["To"].map(lambda s: pdb2res[s] if s in pdb2res else np.nan)
        id_mapping_df = id_mapping_df.sort_values("resolution", ascending=True).drop_duplicates("From")
        id_mapping_df = id_mapping_df.rename({"From": "Uniprot_ID", "To": "pdb_id"}, axis=1).set_index("Uniprot_ID").dropna()
        
        info_df_with_res = self.info_df.set_index("Uniprot_ID").join(id_mapping_df, how="inner")
        print(info_df_with_res)

        pcba_entries: pd.DataFrame = pd.read_csv("/scratch/sx801/scripts/DiffDock-NMDN/scripts/ds_prepare/yaowen_nonbinders/lit_pcba_entries.tsv", sep="\t")
        pcba_uniprot_ids = set(pcba_entries["Entry"].values.reshape(-1).tolist())
        filtered_entries = []
        for i in tqdm(range(info_df_with_res.shape[0])):
            this_info = info_df_with_res.iloc[[i]]
            if this_info.index.item() in pcba_uniprot_ids: continue
            filtered_entries.append(this_info)
        filtered_entries = pd.concat(filtered_entries, axis=0)
        filtered_entries.to_csv(save_csv)
        return filtered_entries


    @cached_property
    def info_df(self) -> pd.DataFrame:
        INTERESED_COLS = ["Activity", "SMILES", "Uniprot_ID"]
        ic50_csv = osp.join(self.ds_root, "ic50.csv")
        ic50_df = pd.read_csv(ic50_csv)
        ic50_df = ic50_df[ic50_df["Activity"] <= self.threshold][INTERESED_COLS]
        ic50_df["file_handle"] = ic50_df.index.map(lambda i: f"nonbinders.ic50.{i}")

        ki_csv = osp.join(self.ds_root, "ki.csv")
        ki_df = pd.read_csv(ki_csv)
        ki_df = ki_df[ki_df["Activity"] <= self.threshold][INTERESED_COLS]
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
        save_root = "/scratch/sx801/temp/DiffDock_crystal_nonbinders_jobs"
        n_gpus = 40
        all_job_info = defaultdict(lambda: [])
        for i in range(self.sampled_pl_info_df.shape[0]):
            this_info = self.sampled_pl_info_df.iloc[i]
            all_job_info["complex_name"].append(this_info["file_handle"])
            pdb_id = this_info["pdb_id"]
            prot_path = self.pdb2prot_og(pdb_id)
            assert osp.exists(prot_path), prot_path
            all_job_info["protein_path"].append(prot_path)
            all_job_info["ligand_description"].append(this_info["SMILES"])
            all_job_info["protein_sequence"].append("")
        all_job_df = pd.DataFrame(all_job_info).sample(frac=1.)
        for i, df_chunk in enumerate(np.array_split(all_job_df, n_gpus)):
            df_chunk.to_csv(osp.join(save_root, f"job_gpu_{i}.csv"), index=False)

if __name__ == "__main__":
    print(NonbinderReader().generate_diffdock_jobs())
