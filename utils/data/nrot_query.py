import pandas as pd
import numpy as np

from utils.configs import Config
from utils.data.rmsd_info_query import CSV_Query


class NRotQuery(CSV_Query):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)

    def get_info_csv(self) -> str:
        return self.cfg.data.pre_computed.nrot_csv
    
    def get_query_key(self) -> str:
        return "n_rotatable_bond"
    
    def get_fill_value(self) -> int:
        csv = "/scratch/sx801/cache/rmsd_csv/pdbbind_yaowen_nonbinders.nrot.csv"
        df = pd.read_csv(csv).dropna()
        return int(np.median(df["n_rotatable_bond"].values))