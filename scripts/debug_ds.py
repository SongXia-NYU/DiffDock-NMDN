from utils.data.DummyIMDataset import DummyIMDataset
from torch_geometric.loader import DataLoader
import pickle
from tqdm import tqdm
import torch


def debug_ds():
    ds = DummyIMDataset(data_root="/scratch/sx801/data/im_datasets/", dataset_name="frag20-ultimate-sol-mmff-04182022.pyg", config_args=None)
    for i in tqdm(range(0, len(ds)), total=len(ds)):
        if ds[i].CalcSol.shape[0] == 0:
            print(i)

    exit
    
    dl = DataLoader(ds[565893:], batch_size=2, pin_memory=True, num_workers=0)
    i = 565893
    for batch in dl:
        print(i)
        i+=2

def fix_split():
    split_f = "/scratch/sx801/data/im_datasets/processed/frag20-ultimate-sol-split-mmff-04182022.pyg"
    split = torch.load(split_f)
    err_idx = set([565896, 605715])
    for split_name in split:
        split_val = split[split_name]
        if split_val is None:
            continue
        fixed_idx = list(set(split_val).difference(err_idx))
        split[split_name] = torch.as_tensor(fixed_idx)
    breakpoint()
    torch.save(split, split_f)

if __name__ == "__main__":
    fix_split()
