import pandas as pd
from glob import glob

class LinF9Query:
    def __init__(self, cfg, query_key: str) -> None:
        self.cfg = cfg
        self.linf9_csv = cfg["linf9_csv"]
        self.query_key = query_key

        linf9_csvs = glob(self.linf9_csv)
        linf9_dfs = [pd.read_csv(csv) for csv in linf9_csvs]
        linf9_dfs = pd.concat(linf9_dfs)
        self.is_casf_screening = False
        if "tgt_pdb" in linf9_dfs.columns and "lig_pdb" in linf9_dfs.columns:
            self.is_casf_screening = True
            linf9_dfs = linf9_dfs.rename({"nmdn_linf9_score": "linf9_score"}, axis=1)
        if "fl" in linf9_dfs.columns:
            linf9_dfs = linf9_dfs.rename({"fl": "file_handle"}, axis=1)
        fl2linf9 = {}
        for i in range(len(linf9_dfs)):
            this_info = linf9_dfs.iloc[i]
            if not self.is_casf_screening: 
                fl = this_info[self.query_key]
            else:
                fl = f"{this_info['tgt_pdb']}_{this_info['lig_pdb']}"
            linf9 = this_info["linf9_score"]
            fl2linf9[fl] = linf9
        self.fl2linf9 = fl2linf9
        self.is_merck_fep = "/merck-fep/" in self.linf9_csv

        # count the stats
        self.query_count: int = 0
        self.missing_count: int = 0
    
    def query_linf9(self, query: str):
        if self.is_casf_screening:
            query = query.split(".")[0]
        if self.is_merck_fep:
            query = ".".join(query.split(".")[:-1])
        self.query_count += 1
         # if too many ligands are missing RMSD values, something went wrong
        if self.query_count >= 1_000:
            missing_percent: float = 1.0 * self.missing_count / self.query_count
            assert missing_percent < 0.5, "Too many missing RMSDs: {:.1f}%".format(100*missing_percent)
        if query not in self.fl2linf9:
            self.missing_count += 1
            return None
        return self.fl2linf9[query]
