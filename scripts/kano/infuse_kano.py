import pickle
import numpy as np
import os
import os.path as osp
import torch
from typing import Dict

embed_f: str = "/scratch/sx801/scripts/KANO/initial/ele2emb.pkl"
kg_ele_embed: Dict[int, np.ndarray] = pickle.load(open(embed_f, "rb"))

srcdir = "exp_pl_369_run_2023-06-12_205926__096246"
dstdir = "exp_pl_369_kano_run_2023-06-12_205926__096246"

src_model_file = osp.join(srcdir, "best_model.pt")
state_dict: dict = torch.load(src_model_file, map_location="cpu")
embed_key = "module.embedding_layer.embedding.weight"
embed: torch.Tensor = state_dict[embed_key]

for i in range(embed.shape[0]):
    kg_embed = kg_ele_embed[i]
    kg_embed_dim = kg_embed.shape[0]
    embed[i, :kg_embed_dim] = torch.as_tensor(kg_embed)
state_dict[embed_key] = embed
os.makedirs(dstdir, exist_ok=True)
torch.save(state_dict, osp.join(dstdir, "best_model.pt"))
