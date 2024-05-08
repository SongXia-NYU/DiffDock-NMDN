from collections import defaultdict
import os.path as osp
import pandas as pd
import torch
from typing import Set

# -----------------You need to modify the following variables----------------- #
# this is the name of the data set
DS_NAME = "DeepAccNet_test"
# this is where you save the running results
SAVE_ROOT = "/scratch/sx801/temp/data"
# -----------------------------------------------------------------------------#

SAVE_ROOT = osp.join(SAVE_ROOT, "processed")
pyg_file = osp.join(SAVE_ROOT, f"{DS_NAME}.pyg")

pdb2split_name = {}
for split_name in ["train", "valid", "test"]:
    split_file = f"/scratch/projects/yzlab/xd638/Protein_MQA/prepared_for_Song/DeepAccNet/lists/list_{split_name}_successful.txt"
    pdb_df = pd.read_csv(split_file, header=None)
    pdbs: Set[str] = set(pdb_df.values.reshape(-1).tolist())
    for pdb in pdbs:
        pdb: str = pdb.split("_protein")[0]
        pdb2split_name[pdb] = split_name

pyg = torch.load(pyg_file)[0]
train_ds_pdbs = pyg.pdb
split_index = defaultdict(lambda: [])
for i, pdb in enumerate(train_ds_pdbs):
    split_name: str = pdb2split_name[pdb]
    split_index[f"{split_name}_index"].append(i)

for split_name in split_index:
    split_index[split_name] = torch.as_tensor(split_index[split_name])
split_index["val_index"] = split_index["valid_index"]

save_split_name = osp.join(SAVE_ROOT, f"{DS_NAME}.split.pth")
torch.save(dict(split_index), save_split_name)
