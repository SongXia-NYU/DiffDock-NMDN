from collections import defaultdict
import pickle
from typing import Dict

import torch
from torch_scatter import scatter_add
import tqdm
import yaml
import os.path as osp
from torch_geometric.loader import DataLoader
from utils.LossFn import calculate_probablity, mdn_loss_fn
from utils.data.prot_embedding_ds import ProteinEmbeddingDS
from utils.eval.tester import Tester
from utils.utils_functions import get_device


class MDNExtractor(Tester):
    def __init__(self, prot_seq_info=None, embed_save_f=None, **kwargs):
        super().__init__(**kwargs)
        self.prot_seq_info = prot_seq_info
        self.embed_save_f = embed_save_f
        
        # only used in self.generate_mdn_embedding()
        self._pdb2seq: Dict[str, str] = None
        self._pl_mapper: dict = None
        self._aa2id: Dict[str, int] = None
        self._atom_num2id: torch.Tensor = None

    def run_test(self):
        return self.generate_mdn_embedding()

    def generate_mdn_embedding(self, ds: ProteinEmbeddingDS = None):
        if ds is None:
            # sometimes dataset is too large to fit into memory. 
            # the dataset is split into chunks of small datasets
            if isinstance(self.ds_test, list):
                for ds_cls, ds_args in self.ds_test:
                    this_ds = ds_cls(**ds_args)
                    self.generate_mdn_embedding(this_ds)
            assert isinstance(self.ds_test, ProteinEmbeddingDS), self.ds_test.__class__
            return self.generate_mdn_embedding(self.ds_test)
        
        out_dict = defaultdict(lambda: [])
        if ds.dataset_name.startswith("PBind2020OG"):
            train_index = ds.train_index
            val_index = ds.val_index
            all_index = torch.concat([train_index, val_index])
            ds = ds[all_index]
            out_dict["train_index"] = torch.arange(train_index.shape[0])
            out_dict["val_index"] = torch.arange(train_index.shape[0], train_index.shape[0] + val_index.shape[0])
        
        # self._pdb2seq is to get the sequence information of each protein, 
        # which is ultimately used to identify the AA type
        if not self._pdb2seq:
            with open(self.prot_seq_info) as f:
                self._pdb2seq = yaml.safe_load(f)

        # self._pl_mapper contains the instruction to map protein AAs and ligand atoms into consecutive numbers
        if not self._pl_mapper:
            with open(osp.join(osp.dirname(__file__), "..", "tables", "mdn_embed_info.yaml")) as f:
                self._pl_mapper = yaml.safe_load(f)
            # map amino acid charater to AA_ID (0-19)
            self._aa2id = self._pl_mapper["prot_aa2id"]
            self._n_aa_types = len(self._pl_mapper["prot_aa2id"])
            # map atomic number into ATOM_ID (0-14), 0 for others.
            self._atom_num2id = torch.zeros(95).long()
            for atom_num in self._pl_mapper["atom_num2id"]:
                self._atom_num2id[atom_num] = self._pl_mapper["atom_num2id"][atom_num]
            self._n_atom_types = 1 + len(self._pl_mapper["atom_num2id"])
            self._embed_dim_size = self._n_aa_types * self._n_atom_types

        # batch_size has to be one!
        dl: DataLoader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=self.num_workers)
        if self.use_tqdm:
            dl = tqdm.tqdm(dl, desc="generate MDN", total=len(ds))
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dl):
                batch = batch.to(get_device())
                out = self.model(batch)
                pair_nll =  mdn_loss_fn(out["pi"], out["sigma"], out["mu"], out["dist"])
                pair_prob = calculate_probablity(out["pi"], out["sigma"], out["mu"], out["dist"])

                prot_seq = ""
                for idx in batch.sample_id:
                    pdb = ds.idx2pdb_mapper[idx.item()]
                    prot_seq += self._pdb2seq[pdb]
                prot_aa_id = torch.as_tensor([self._aa2id[aa] for aa in prot_seq])[out["pl_edge_index_used"][1]].to(pair_nll.device)
                lig_atom_num = batch.Z[out["pl_edge_index_used"][0]]
                lig_atom_id = self._atom_num2id[lig_atom_num].to(pair_nll.device)
                # the pair_index ranges from 0 to (N_PROT_TYPES * N_LIG_TYPES -1)
                pl_pair_index = prot_aa_id * self._n_atom_types + lig_atom_id
                
                nll_embed = scatter_add(pair_nll, pl_pair_index, dim_size=self._embed_dim_size)
                out_dict["nll_embed"].append(nll_embed)
                prob_embed = scatter_add(pair_prob, pl_pair_index, dim_size=self._embed_dim_size)
                out_dict["prob_embed"].append(prob_embed)
                out_dict["protein_file"].append(batch.protein_file[0])
                out_dict["ligand_file"].append(batch.ligand_file[0])
                for prop_name in ["pKd", "pdb", "linf9"]:
                    if not hasattr(batch, prop_name):
                        continue
                    out_dict[prop_name].append(getattr(batch, prop_name)[0])
        with open(osp.join(self.ds.root, "processed", self.embed_save_f), "wb") as f:
            pickle.dump(dict(out_dict), f)
            