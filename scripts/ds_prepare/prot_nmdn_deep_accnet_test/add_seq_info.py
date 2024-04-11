import copy
import torch
from tqdm import tqdm

from geometry_processors.pl_dataset.prot_utils import pdb2seq

ds_path = "/scratch/sx801/data/im_datasets/processed/DeepAccNet_testset.pyg"
data, slices = torch.load(ds_path)
seq_list = []
for protf in tqdm(data.protein_file):
    seq = pdb2seq(protf)
    seq_list.append(seq)
data.seq = seq_list
slices["seq"] = copy.deepcopy(slices["protein_file"])
torch.save((data, slices), ds_path)
