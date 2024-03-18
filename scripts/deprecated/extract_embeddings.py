import os

import torch
from torch_scatter import scatter
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from Networks.PhysDimeNet import PhysDimeNet
from test import read_folder_config, get_test_set
from utils.utils_functions import get_device, floating_type


def extract_embeddings(folder_name):
    args, config_name = read_folder_config(folder_name)
    args["requires_atom_embedding"] = True
    args["requires_atom_prop"] = True

    net = PhysDimeNet(cfg=args, **args)
    net = net.to(get_device())
    net = net.type(floating_type)
    model_data = torch.load(os.path.join(folder_name, 'best_model.pt'), map_location=get_device())
    incompatible = net.load_state_dict(model_data)
    print(incompatible)

    data_provider = get_test_set(args)
    data_loader = DataLoader(data_provider, shuffle=False, batch_size=1)

    embedding = []
    embedding_ss = []
    activity = []
    mol_prop = []
    exp_dft_diff = []

    for data in tqdm(data_loader):
        out = net(data.to(get_device()))
        embedding.append(scatter(out["atom_embedding"], out["atom_mol_batch"], reduce="sum", dim=0).detach().cpu())
        embedding_ss.append(scatter(out["atom_embedding_ss"], out["atom_mol_batch"], reduce="sum", dim=0).detach().cpu())
        activity.append(data.activity.detach().cpu())
        mol_prop.append(out["mol_prop"].detach().cpu())
        exp_dft_diff.append((data.activity - data.CalcLogP).cpu())
    embedding_dataset = {
        "embedding": torch.cat(embedding, dim=0),
        "embedding_ss": torch.cat(embedding_ss, dim=0),
        "activity": torch.cat(activity, dim=0),
        "mol_prop": torch.cat(mol_prop, dim=0),
        "exp_dft_diff": torch.cat(exp_dft_diff, dim=0)
    }
    torch.save(embedding_dataset, "lipop_embed160_exp339_qm_diff_dataset.pt")
    print("finished")


if __name__ == '__main__':
    extract_embeddings("../exp339_run_2021-08-16_121007")
