import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob


def multi_run_avg(folders):
    result_csv = pd.DataFrame()
    for folder in glob(folders):
        this_folder_data = {"folder": folder}
        for split in ["val", "test"]:
            file_pt = glob(osp.join(folder, "*_test_*", f"loss_*_{split}.pt"))
            assert len(file_pt) == 1
            file_pt = file_pt[0]
            data = torch.load(file_pt)
            for key in data.keys():
                if isinstance(data[key], float):
                    this_folder_data[f"{split.upper()}_{key}"] = data[key]
        result_csv = result_csv.append(this_folder_data, ignore_index=True)
    save_base = osp.basename(glob(folders)[0]).split("-")[0]+"-rand"
    result_csv.to_csv(save_base + ".csv")
    with open(save_base+".txt", "w") as f:
        test_data = result_csv["TEST_RMSE_activity"].values
        val_data = result_csv["VAL_RMSE_activity"].values
        f.write(f"VAL RMSE {val_data.mean()} +- {val_data.std()} \n")
        f.write(f"TEST RMSE {test_data.mean()} +- {test_data.std()} \n")


if __name__ == '__main__':
    multi_run_avg("../../tmp/rand_results/exp320-rand*_run_*")
