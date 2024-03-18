import torch
import os.path as osp
import pickle
import yaml
from typing import Set
import numpy as np

def check_ds_overlap(query_pdbs: Set[str], print_overlap: bool = False):
    print("********")
    print(f"Total number of queried {len(query_pdbs)}")
    # PDBBind 2020 filtered data set
    dsroot = "/scratch/sx801/data/im_datasets/processed"
    ds_file: str = osp.join(dsroot, "PBind2020OG.polar.polar.implicit.min_dist.linf9.pyg")
    split_file: str = osp.join(dsroot, "PBind2020OG.polar.polar.implicit.min_dist.crysonly.covbinder.res2.5.testsets.poslinf9.pth")
    ds = torch.load(ds_file)
    split = torch.load(split_file)
    # ---> train and valid pdbs
    pbind_train_pdbs: Set[str] = set([ds[0].pdb[i] for i in split["train_index"]])
    pbind_val_pdbs: Set[str] = set([ds[0].pdb[i] for i in split["val_index"]])
    del ds, split
    print("********")
    print(f"Total number of training PDBs in PDBBind2020: {len(pbind_train_pdbs)}")
    print(f"Number of overlaps: {len(query_pdbs.intersection(pbind_train_pdbs))}")
    if print_overlap: print(f"Overlapping PDBs: {query_pdbs.intersection(pbind_train_pdbs)}")
    print("----")
    print(f"Total number of validation PDBs in PDBBind2020: {len(pbind_val_pdbs)}")
    print(f"Number of overlaps: {len(query_pdbs.intersection(pbind_val_pdbs))}")
    if print_overlap: print(f"Overlapping PDBs: {query_pdbs.intersection(pbind_val_pdbs)}")
    print("********")

    # DiffDock filtered data set
    diffdock_pdbs_yaml: str = "/scratch/sx801/scripts/DiffDock/data/splits/" + \
        "PBind2020.linf9.covbinder.res2.5.testsetsv3.poslinf9.yaml"
    with open(diffdock_pdbs_yaml) as f:
        pdbs = yaml.safe_load(f)
    ddock_pdbs: Set[str] = set(pdbs)
    print(f"Total number of PDBs in DiffDock train/val set: {len(ddock_pdbs)}")
    print(f"Number of overlaps: {len(query_pdbs.intersection(ddock_pdbs))}")
    if print_overlap: print(f"Overlapping PDBs: {query_pdbs.intersection(ddock_pdbs)}")
    print("********")

    # GenScore train and validation
    genscore_ids_file: str = f"/vast/sx801/geometries/rtmscore_s/v2020_train_ids.npy"
    genscore_ids: Set[str] = set(np.load(genscore_ids_file)[0].tolist())
    print(f"Total number of PDBs in GenScore train/val set: {len(genscore_ids)}")
    print(f"Number of overlaps: {len(query_pdbs.intersection(genscore_ids))}")
    if print_overlap: print(f"Overlapping PDBs: {query_pdbs.intersection(genscore_ids)}")
    print("********")


if __name__ == "__main__":
    check_ds_overlap(set(["5hnb", "3l9h", "5tbm", "5ehr", "4ui5", "4r1y", "6hvi", "4pv0"]), True)
