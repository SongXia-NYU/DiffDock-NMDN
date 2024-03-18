from collections import defaultdict
import json
import logging
import os
import os.path as osp
from glob import glob
from typing import Dict, List, Union
from tqdm import tqdm
import pandas as pd

import torch
from prody import parsePDB
import esm
from esm.pretrained import load_model_and_alphabet_hub
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from geometry_processors.pl_dataset.prot_utils import pdb2seq

class ESMCalculator:
    def __init__(self, save_root, model_name="esm2_t33_650M_UR50D", repr_layer=33) -> None:
        self.save_root = save_root
        self.model_name = model_name
        self.repr_layer = repr_layer
        self.n_fail: int = 0
        self.n_success: int = 0

        self._model = None
        self._batch_converter = None

    def run(self, seq_info: Dict[str, str]):
        """
        seq_info: a dictionary, key is pdb_id, value is sequence (1 character representation of AAs)
        """
        os.makedirs(self.save_root, exist_ok=True)
        for pdb_id in tqdm(seq_info.keys()):
            save_path = osp.join(self.save_root, f"{pdb_id}.pth")
            if osp.exists(save_path) or seq_info[pdb_id] is None:
                continue

            sequences :Union[str, List[str]] = seq_info[pdb_id]
            if isinstance(sequences, str):
                sequences: List[str] = [sequences]
            token_representations: List[torch.Tensor] = [self.embed_from_seq(seq) for seq in sequences]
            if None in token_representations:
                print("Error proc ", pdb_id)
                continue
            token_representations = self.trim_reprs(token_representations)
            token_representations = torch.concat(token_representations, dim=1)
            torch.save(token_representations, save_path)

    def run_chunks(self, seq_info: Dict[str, Union[str, List[str]]], chunksize=4000):
        """
        seq_info: a dictionary, key is pdb_id, value is sequence (1 character representation of AAs)
        """
        self.n_fail, self.n_success = 0, 0
        os.makedirs(self.save_root, exist_ok=True)
        this_chunk_embed = {}
        chunk2pdb = defaultdict(lambda: [])

        for i, pdb_id in tqdm(enumerate(seq_info.keys()), total=len(list(seq_info.keys()))):
            chunk_id = i // chunksize
            chunk2pdb[chunk_id].append(pdb_id)

            save_chunk = osp.join(self.save_root, f"chunk_{chunk_id}.pth")
            # skipping checkpoint if exists
            fakerun = osp.exists(save_chunk)
            if fakerun or seq_info[pdb_id] is None:
                continue

            sequences :Union[str, List[str]] = seq_info[pdb_id]
            if isinstance(sequences, str):
                sequences: List[str] = [sequences]

            token_representations: List[torch.Tensor] = [self.embed_from_seq(seq) for seq in sequences]
            if None in token_representations:
                print("Error proc ", pdb_id)
                continue
            token_representations = self.trim_reprs(token_representations)
            token_representations = torch.concat(token_representations, dim=1)
            this_chunk_embed[pdb_id] = token_representations

            if (i+1) % chunksize == 0:
                torch.save(this_chunk_embed, save_chunk)
                this_chunk_embed = {}

        save_chunk = osp.join(self.save_root, f"chunk_{chunk_id}.pth")
        fakerun = osp.exists(save_chunk)
        if not fakerun and this_chunk_embed:
            torch.save(this_chunk_embed, save_chunk)

        torch.save(dict(chunk2pdb), osp.join(self.save_root, "chunk2pdb.pth"))
        print("-----------Completed-----------")
        print("#Failed: ", self.n_fail)
        print("#Success: ", self.n_success)

    def trim_reprs(self, reprs: List[torch.Tensor]):
        if len(reprs) == 1:
            return reprs
        
        # a special logic to trim the reprentations for downstream code
        # the first and last token are appended to each sequence, so they have to be removed
        # to obtain proper sequence embedding. The downstream code is going to deal with that,
        # but the concatenation will mess that up. The following scripts help to solve the problem.
        reprs[0] = reprs[0][:, 1:, :]
        reprs[-1] = reprs[-1][:, :-1, :]
        for i in range(1, len(reprs)-1):
            reprs[i] = reprs[i][:, 1: -1, :]
        return reprs


    def embed_from_seq(self, seq: str):
        this_data = [("protein", seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(this_data)
        with torch.no_grad():
            try:
                results = self.model(batch_tokens.cuda(), repr_layers=[self.repr_layer], return_contacts=True)
            except RuntimeError as e:
                self.n_fail += 1
                print(e)
                return None
        token_representations = results["representations"][self.repr_layer]
        self.n_success += 1
        return token_representations

    @property
    def batch_converter(self):
        if self._batch_converter is not None:
            return self._batch_converter
        
        __ = self.model
        return self._batch_converter

    @property
    def model(self):
        if self._model is not None:
            return self._model
        
        # init the distributed world with world_size 1
        url = "tcp://localhost:23456"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

        torch_hub_cache = "/scratch/sx801/scripts/DiffDock/data/downloaded_model"
        os.makedirs(torch_hub_cache, exist_ok=True)
        torch.hub.set_dir(torch_hub_cache)

        # Load ESM-2 model
        model, alphabet = load_model_and_alphabet_hub(self.model_name)
        # model = model.cuda()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        # initialize the model with FSDP wrapper
        fsdp_params = dict(
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            cpu_offload=True,  # enable cpu offloading
        )
        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            model = wrap(model)
        self._model = model
        self._batch_converter = batch_converter
        return self._model
