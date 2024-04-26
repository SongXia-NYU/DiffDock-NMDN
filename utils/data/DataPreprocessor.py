from typing import Any, Union

from utils.configs import Config
from utils.data.MyData import MyData
from torch_geometric.data import HeteroData


class DataPreprocessor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        if cfg is None:
            return
        self.proc_lit_pcba = cfg.data.proc_lit_pcba

    def __call__(self, data: Union[MyData, HeteroData], idx: int) -> Union[MyData, HeteroData]:
        if self.cfg is None:
            return data
        
        if self.proc_lit_pcba:
            file_handle: str = data.file_handle
            rank: int = data.rank.cpu().item()
            file_handle_ranked = f"{file_handle}.rank{rank}"
            data.file_handle_ranked = file_handle_ranked
        return data
