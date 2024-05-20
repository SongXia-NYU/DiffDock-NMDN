import os.path as osp
import torch
import pandas as pd

class SaveRootQuery:
    def __init__(self, save_root: str) -> None:
        self.save_root = save_root
        self.cache = {}
    
    def query(self, query: str):
        if query in self.cache:
            return query
        
        data = self.load_data(query)
        self.cache[query] = data
        return data
    
    def load_data(self, query: str):
        raise NotImplementedError
    
class AtomSolQuery(SaveRootQuery):
    def load_data(self, query: str):
        if not osp.exists(osp.join(self.save_root, f"{query}.pth")):
            return None
        
        return torch.load(osp.join(self.save_root, f"{query}.pth"))

class SasaQuery(SaveRootQuery):
    def load_data(self, query: str):
        if not osp.exists(osp.join(self.save_root, f"{query}.pkl")):
            return None
        
        return pd.read_pickle(osp.join(self.save_root, f"{query}.pkl"))