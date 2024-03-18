import torch
from torch_geometric.data import Data


from typing import Optional


class MyData(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_data(data: Data):
        if isinstance(data, MyData):
            return data

        kwargs = {}
        # due to some errors in data processing, I have to double check data.keys or data.keys()
        keys = data.keys
        if not isinstance(keys, list):
            keys: list = keys()
        for key in keys:
            val = getattr(data, key)
            if isinstance(val, list) and isinstance(val[0], str) and key != "pdb":
                val = [[v] for v in val]
            kwargs[key] = val
        return MyData(**kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        # protein protein interaction
        if key in ["PP_min_dist_edge_index", "PP_min_dist_oneway_edge_index"]:
            # inter pp interaction
            if hasattr(self, "N_aa_chain1"):
                return torch.as_tensor([[self.N_aa_chain1], [self.N_aa_chain2]])
            # intera-pp interaction
            return self.prot_embed.shape[0]
        if key == "PP_min_dist_chain1_oneway_edge_index":
            return self.N_aa_chain1
        if key == "PP_min_dist_chain2_oneway_edge_index":
            return self.N_aa_chain2
        if key == "PL_min_dist_sep_oneway_edge_index":
            # this index is only used in pad-style dataset, stored as lig_id --> prot_id
            # dim0 is ligand_id and dim1 is protein id
            # only ligand and protein are saved independently
            return torch.as_tensor([[self.N], [self.prot_embed.shape[0]]])
        if "_edge_index" in key:
            return self.N
        if key == "martini_aa_batch":
            return self.N_p_aa
        return super(MyData, self).__inc__(key, value, *args, **kwargs)

    @property
    def num_nodes(self) -> Optional[int]:
        return self.N