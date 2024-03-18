import torch
import torch_geometric
import pandas as pd
import gauss.read_gauss_log
import os.path as osp

from tqdm import tqdm
from rdkit.Chem import RemoveHs, MolFromSmiles, RemoveStereochemistry, MolToSmiles

from geometry_processors.DataPrepareUtils import my_pre_transform


def sdf2pyg(sdf_list=None, info_csv=None, info_df=None, save_path_pyg=None, save_path_split=None, check_valid_mol=True,
            smiles_name="cano_smiles", use_tqdm=True):
    from torch_geometric.data import Data

    if info_df is None:
        info_df = pd.read_csv(info_csv)
    split = {"train_index": [],
             "valid_index": [],
             "test_index": []}
    data_list = []
    info = enumerate(sdf_list)
    if use_tqdm:
        info = tqdm(info, total=len(sdf_list))
    for i, sdf in info:
        this_info: dict = info_df.iloc[i].to_dict()
        sdf_info = gauss.read_gauss_log.Gauss16Log(log_path=None, log_sdf=sdf, supress_warning=True)

        if check_valid_mol:
            # assert smiles1 == smiles2
            mol1 = RemoveHs(sdf_info.mol)
            mol2 = RemoveHs(MolFromSmiles(this_info[smiles_name]))
            RemoveStereochemistry(mol1)
            RemoveStereochemistry(mol2)
            smiles1 = MolToSmiles(mol1)
            smiles2 = MolToSmiles(mol2)
            assert smiles1 == smiles2, f"{smiles1} \n {smiles2}"

        gauss.read_gauss_log.Gauss16Log.conv_type(this_info)

        this_info.update(sdf_info.get_basic_dict())

        this_data = Data(**this_info)
        this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                     cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                     bond_atom_sep=False, record_long_range=True)
        data_list.append(this_data)

        group = this_info["group"] if "group" in this_info else "test"
        split[f"{group}_index"].append(i)

    torch.save(torch_geometric.data.InMemoryDataset.collate(data_list), save_path_pyg)
    if save_path_split is not None:
        torch.save(split, save_path_split)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_list", type=str, nargs="*")
    parser.add_argument("--info_csv", type=str)
    parser.add_argument("--save_path_pyg", type=str)
    parser.add_argument("--save_path_split", type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    sdf2pyg(**args)


def main_test():
    from glob import glob
    sdf_list = glob("../data/lipop_sol/lipop_mmff_sdfs/*.mmff.sdf")
    sdf_list.sort(key=lambda x: int(osp.basename(x).split(".")[0]))
    args = {
        "sdf_list": sdf_list,
        "info_csv": "../data/lipop_sol/lipop_paper.csv",
        "save_path_pyg": "../data/lipop_sol/lipop_mmff_pyg.pt",
        "save_path_split": "../data/lipop_sol/lipop_mmff_pyg_split.pt"
    }

    sdf2pyg(**args)


if __name__ == '__main__':
    main()
