import torch

if False:
    d = torch.load("/scratch/sx801/data/im_datasets/processed/PL_train-prot.polar.lig.polar.pyg")

    print(d[0].source[:10])
    print(d[0].ligand_file[:10])
    print(d[0].protein_file[:10])

import os.path as osp
import pandas as pd
info_csv = osp.join("/scratch/sx801/data", "pl_xgb_lin_f9.csv")
info_mapper = pd.read_csv(info_csv).set_index("lig_name")["lin_f9"].to_dict()
print(info_mapper)