from typing import Union
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob

from utils.configs import Config
from utils.utils_functions import lazy_property


class CSV_Query:
    # Query pre-computed values saved in a/multiple CSV file(s)
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.info_csv: str = self.get_info_csv()
        # count the stats
        self.query_count: int = 0
        self.missing_count: int = 0
        self.lig_identifier_dst: str = cfg.data.pre_computed["lig_identifier_dst"]
        if "casf.docking.rmsd.csv" in self.info_csv:
            self.lig_identifier_dst = "pdb"

    def get_info_csv(self) -> str:
        raise NotImplementedError
    
    def get_query_key(self) -> str:
        raise NotImplementedError

    @lazy_property
    def fill_value(self) -> Union[float, int]:
        return self.get_fill_value()

    def get_fill_value(self) -> Union[float, int]:
        raise NotImplementedError

    def query(self, query: str) -> Union[float, int]:
        # more efficient query
        if self.lig_identifier_dst == "file_handle_ranked":
            file_handle, rank = query.split(".rank")
            file_handle = int(file_handle.split("_")[0])
            rank = int(rank)
            query = (file_handle, rank)
        self.query_count += 1
        if query in self.info_df.index:
            try:
                rmsd_info = self.info_df.loc[query, self.get_query_key()]
                if isinstance(rmsd_info, pd.Series):
                    rmsd_info = rmsd_info.item()
                if rmsd_info is not None and not np.isnan(rmsd_info):
                    return rmsd_info
            except ValueError:
                pass
        self.missing_count += 1
        # if too many ligands are missing RMSD values, something went wrong
        if self.query_count >= 1_000:
            missing_percent: float = 1.0 * self.missing_count / self.query_count
            assert missing_percent < 0.2, "Too many missing {}s: {:.1f}%".format(self.get_query_key(), 100*missing_percent)
        return self.fill_value

    @lazy_property
    def info_df(self) -> pd.DataFrame:
        rmsd_csvs = glob(self.info_csv)
        if len(rmsd_csvs) == 1:
            return pd.read_csv(rmsd_csvs[0]).set_index(self.lig_identifier_dst)
        
        # concat multiple csvs
        data_frames = [pd.read_csv(csv).set_index(self.lig_identifier_dst) for csv in rmsd_csvs]
        rmsd_df = pd.concat(data_frames, axis=0)
        rmsd_df = rmsd_df[~rmsd_df.index.duplicated(keep='first')]
        # more efficient query
        if self.lig_identifier_dst == "file_handle_ranked":
            rmsd_df = rmsd_df.reset_index()
            rmsd_df["file_handle"] = rmsd_df["file_handle_ranked"].map(lambda s: int(s.split("_")[0]))
            rmsd_df["rank"] = rmsd_df["file_handle_ranked"].map(lambda s: int(s.split(".rank")[1]))
            rmsd_df = rmsd_df.set_index(["file_handle", "rank"]).sort_index()
        return rmsd_df


class RMSD_Query(CSV_Query):
    # Query pre-computed RMSD information 
    # The pre-computed RMSD is used as one of the feature to predict pKd.
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_info_csv(self) -> str:
        return self.cfg.data.pre_computed["rmsd_csv"]

    def get_query_key(self) -> str:
        return "rmsd"

    def get_fill_value(self) -> float:
        # use training set RMSD
        training_df = pd.read_csv("/vast/sx801/geometries/PL_physics_infusion/PDBBind2020_OG/info/pdbbind2020_og.rmsd.csv")
        return np.median(training_df["rmsd"].values)
