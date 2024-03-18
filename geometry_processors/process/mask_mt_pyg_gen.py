import argparse

import pandas as pd
import os.path as osp

import torch
import torch_geometric.data

from geometry_processors.calc_sol import logP_to_watOct

PROPS = ["gasEnergy", "watEnergy", "octEnergy", "CalcSol", "CalcOct", "watOct"]


def mask_mt_pyg_gen(intersection_csv, ds1_pygs, ds2_pygs, csv1, csv2, pyg_no_aux, pyg_aux):
    intersection_df = pd.read_csv(intersection_csv, dtype={"sample_id_1": int, "sample_id_2": int})
    ids_1 = set(intersection_df["sample_id_1"].values.tolist())
    ids_2 = set(intersection_df["sample_id_2"].values.tolist())

    def _load_csv(csv):
        df = pd.read_csv(csv).rename({"FileHandle": "sample_id", "Kow": "activity"}, axis=1)
        return df.astype({"sample_id": int}).set_index("sample_id")
    df1 = _load_csv(csv1)
    df2 = _load_csv(csv2)

    data_list = []

    intersection_df = intersection_df.set_index("sample_id_1")
    for pyg_f in ds1_pygs:
        try:
            pyg = torch.load(pyg_f)

            sample_id = int(osp.basename(pyg_f).split(".")[0])
            if sample_id in ids_1:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([intersection_df.loc[sample_id][key]]))
                pyg.mask = torch.as_tensor([1, 1, 1, 1, 1, 1]).bool().view(1, -1)
            else:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([9999.]))
                setattr(pyg, "CalcSol", torch.as_tensor([df1.loc[sample_id]["activity"]]))
                pyg.mask = torch.as_tensor([0, 0, 0, 1, 0, 0]).bool().view(1, -1)

            data_list.append(pyg)
        except Exception as e:
            print(f"Error processing {pyg_f}: {e}")

    res = torch_geometric.data.InMemoryDataset.collate(data_list)
    torch.save(res, pyg_no_aux)

    for pyg_f in ds2_pygs:
        try:
            pyg = torch.load(pyg_f)

            sample_id = int(osp.basename(pyg_f).split(".")[0])
            if sample_id not in ids_2:
                for key in PROPS:
                    setattr(pyg, key, torch.as_tensor([9999.]))
                setattr(pyg, "watOct", torch.as_tensor([df2.loc[sample_id]["activity"] * logP_to_watOct]))
                pyg.mask = torch.as_tensor([0, 0, 0, 0, 0, 1]).bool().view(1, -1)
                data_list.append(pyg)
        except Exception as e:
            print(f"Error processing {pyg_f}: {e}")

    res = torch_geometric.data.InMemoryDataset.collate(data_list)
    torch.save(res, pyg_aux)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection_csv")
    parser.add_argument("--ds1_pygs")
    parser.add_argument("--ds2_pygs")
    parser.add_argument("--csv1")
    parser.add_argument("--csv2")
    parser.add_argument("--pyg_aux")
    parser.add_argument("--pyg_no_aux")
    args = parser.parse_args()
    args = vars(args)

    for name in ["ds1_pygs", "ds2_pygs"]:
        with open(args[name]) as f:
            args[name] = f.read().split()

    mask_mt_pyg_gen(**args)


if __name__ == '__main__':
    main()

