import torch
import matplotlib.pyplot as plt
from active_learning_vis import ALFolder
import os.path as osp


def ensemble_curve(root_folder):
    ens_val_mae = []
    ens_test_mae = []
    for i in range(1, 21):
        info = ALFolder(root_folder, first_i=i)
        info_df = info.info_df
        ens_val_mae.append(info_df["val_MAE_E"] * 23.061)
        ens_test_mae.append(info_df["test_MAE_E"] * 23.061)

    plt.figure(figsize=(10, 8))
    plt.xlabel("Number of ensembles")
    plt.ylabel("MAE, kcal/mol")
    plt.title("Test MAE vs. number of ensembles on Frag20_21k")
    plt.xticks(list(range(1, 21)))
    plt.plot(ens_test_mae, marker="o")
    plt.savefig(osp.join(root_folder, "ensemble_curve.png"))
    plt.close()


if __name__ == '__main__':
    ensemble_curve("../../raw_data/exp_misc/exp21k-inall-ensemble20-combined_active_ALL_2021-11-04_150257")
