import argparse
import copy

import pandas as pd
import torch
import torch_geometric.data
import os.path as osp

ev2kcal = 23.06035


def concat_sol(pygs: list,  gas_csvs: list, water_csvs: list, oct_csvs: list, save_pyg: str, save_split: str):
    assert len(pygs) == len(gas_csvs)
    assert len(pygs) == len(water_csvs)
    assert len(pygs) == len(oct_csvs)

    data_list = []
    gas_dfs = []
    water_dfs = []
    oct_dfs = []
    for pyg, gas_csv, water_csv, oct_csv in zip(pygs,  gas_csvs, water_csvs, oct_csvs):
        try:
            pyg = torch.load(pyg)
            gas_df = pd.read_csv(gas_csv)
            gas_df["sample_id"] = osp.basename(gas_csv).split(".")[0]
            gas_dfs.append(gas_df)
            setattr(pyg, "gasEnergy", torch.as_tensor(gas_df["E_atom(eV)"]))

            if water_csv is not None and oct_csv is not None:
                water_df = pd.read_csv(water_csv)
                oct_df = pd.read_csv(oct_csv)
                water_df["sample_id"] = osp.basename(water_csv).split(".")[0]
                oct_df["sample_id"] = osp.basename(oct_csv).split(".")[0]
                water_dfs.append(water_df)
                oct_dfs.append(oct_df)

                water_solv = (water_df["E(eV)"] - gas_df["E(eV)"]) * ev2kcal
                oct_solv = (oct_df["E(eV)"] - gas_df["E(eV)"]) * ev2kcal
                water_oct = (water_df["E(eV)"] - oct_df["E(eV)"]) * ev2kcal

                setattr(pyg, "CalcSol", torch.as_tensor(water_solv))
                setattr(pyg, "CalcOct", torch.as_tensor(oct_solv))
                setattr(pyg, "watOct", torch.as_tensor(water_oct))

                setattr(pyg, "watEnergy", torch.as_tensor(water_df["E_atom(eV)"]))
                setattr(pyg, "octEnergy", torch.as_tensor(oct_df["E_atom(eV)"]))

            data_list.append(pyg)
        except Exception as e:
            print(f"Something is wrong with {gas_csv}: {e}")

    data_concat = torch_geometric.data.InMemoryDataset.collate(data_list)
    torch.save(data_concat, save_pyg)
    pd.concat(gas_dfs).to_csv(osp.join(osp.dirname(save_pyg), "gas_concat.csv"), index=False)
    if water_csvs[0] is not None:
        pd.concat(water_dfs).to_csv(osp.join(osp.dirname(save_pyg), "water_concat.csv"), index=False)
        pd.concat(oct_dfs).to_csv(osp.join(osp.dirname(save_pyg), "oct_concat.csv"), index=False)

    split = {
        "train_index": None,
        "val_index": None,
        "test_index": torch.arange(len(data_list))
    }
    torch.save(split, save_split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pygs")
    parser.add_argument("--gas_csvs")
    parser.add_argument("--water_csvs", default=None)
    parser.add_argument("--oct_csvs", default=None)
    parser.add_argument("--save_pyg")
    parser.add_argument("--save_split")

    args = parser.parse_args()
    args = vars(args)
    processed_args = copy.deepcopy(args)

    num = None
    for name in ["pygs", "gas_csvs", "water_csvs", "oct_csvs"]:
        if args[name] is None:
            processed_args[name] = [None] * num
            continue

        with open(args[name]) as f:
            processed_args[name] = f.read().split()

        if num is None:
            num = len(processed_args[name])

    concat_sol(**processed_args)


if __name__ == '__main__':
    main()
