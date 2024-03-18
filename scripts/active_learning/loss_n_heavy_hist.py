import torch
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import os.path as osp
from torch_scatter import scatter


def main(test_path, loss_fn, name="MAE"):
    for split in ["val", "test"]:
        loss_path = glob(osp.join(test_path, f"loss_*_{split}.pt"))
        assert len(loss_path) == 1
        loss_path = loss_path[0]
        loss_detail = torch.load(loss_path)
        n_heavy_list = scatter((loss_detail["ATOM_Z"] > 1).long(), loss_detail["ATOM_MOL_BATCH"], reduce="sum")
        positions = []
        x = []
        loss = loss_fn(loss_detail)
        for n_heavy in set(n_heavy_list.tolist()):
            positions.append(n_heavy)
            x.append(loss[n_heavy_list == n_heavy])

        plt.boxplot(x, positions=positions)
        plt.xlabel("Num of heavy atoms")
        plt.ylabel(f"{name}, kcal/mol")
        plt.savefig(osp.join(test_path, f"MAE_n_heavy_{split}.png"))
        plt.close()

        x_20 = [loss[n_heavy_list <= 20], loss[n_heavy_list > 20]]
        plt.boxplot(x_20, labels=["n heavy<=20", "n_heavy>20"])
        plt.ylabel(f"{name}, kcal/mol")
        plt.savefig(osp.join(test_path, f"MAE_n_heavy20_{split}.png"))
        plt.close()

        print("hello")


def _cal_sol_diff(loss_detail):
    pred = loss_detail["PROP_PRED"]
    tgt = loss_detail["PROP_TGT"]
    return (pred[:, 4] - tgt[:, 4]).abs()


# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


def _cal_log_p(loss_detail):
    pred = loss_detail["PROP_PRED"]
    tgt = loss_detail["PROP_TGT"]
    return torch.sqrt((pred[:, 5] - tgt[:, 5])**2 )/ logP_to_watOct


if __name__ == '__main__':
    for p in glob("../../tmp/exp*_run_*/exp*_test_*"):
        args = {
            "test_path": p,
            "loss_fn": _cal_log_p,
            "name": "RMSE"
        }
        main(**args)
