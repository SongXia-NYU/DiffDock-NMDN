# prepare ele2emb.pkl into a format readable by main script for metal element embedding
import pickle
import torch

with open("ele2emb.pkl", "rb") as f:
    d_dict = pickle.load(f)

out = []
for ele_num in range(108):
    out.append(torch.as_tensor(d_dict[ele_num]).view(1, -1))
out = torch.concat(out, dim=0)

torch.save(out, "/scratch/sx801/data/im_datasets/processed/kano_kg_atom_embed.pth")