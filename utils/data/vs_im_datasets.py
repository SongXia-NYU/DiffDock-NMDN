import os.path as osp
import numpy as np

from utils.data.DummyIMDataset import VSDummyIMDataset, DummyIMDataset
from torch_geometric.data import Data


class ChunkMapperDataset(VSDummyIMDataset):
    """
    Like VSDummyIMDataset, the protein data is only saved once. This time, the dataset is saved in chunks because they are too large to fit into memory.
    chunk_0.pyg contains the protein information, chunk_1, 2, 3... do not.
    """
    def __init__(self, data_root, dataset_name, config_args, split=None, sub_ref=False, convert_unit=True, valid_size=1000, collate=False, ref=False, **kwargs):
        super().__init__(data_root, dataset_name, config_args, split, sub_ref, convert_unit, valid_size, collate, ref, **kwargs)

        self._protein_mapper = None
        self.basename = osp.basename(dataset_name).split(".pyg")[0]
        if self.basename != "chunk_0":
            chunk0_ds_name = osp.join(osp.dirname(dataset_name), "chunk_0.pyg")
            chunk0_ds = DummyIMDataset(data_root, chunk0_ds_name, config_args)
            self.chunk0_ds = chunk0_ds

        idx2pdb_mapper = {}
        for i, pdb in enumerate(self.data.pdb):
            idx2pdb_mapper[i] = pdb
        self.idx2pdb_mapper = idx2pdb_mapper
    
    @property
    def has_prot_context(self):
        if self._has_prot_context is None:
            self._has_prot_context = set()

            # only chunk_0 has protein information
            # all other chunks do not have protein info
            if self.basename == "chunk_0":
                all_indices = np.arange(len(self))
                self._has_prot_context = set(all_indices.tolist())
        return self._has_prot_context

    def get_d0_di(self, idx):
        assert self.basename != "chunk_0", self.basename
        d0 = self.protein_mapper[self.idx2pdb_mapper[idx]]
        di = super().get(idx, process=False)
        return d0.to(di.R.device), di
    
    @property
    def protein_mapper(self):
        assert self.basename != "chunk_0", self.basename

        if self._protein_mapper is None:
            mapper = {}
            for i in range(len(self.chunk0_ds)):
                this_d = self.chunk0_ds[i]
                mapper[this_d.pdb] = this_d
            self._protein_mapper = mapper
        return self._protein_mapper
