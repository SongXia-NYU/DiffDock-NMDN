import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os.path as osp


def learning_curve():
    root = "../../raw_data/frag20-sol-finals"
    ft_exps = glob(osp.join(root, "exp_ultimate_freeSolv_2*RDrun_*"))
    ft_exps.sort()
    ts_exps = glob(osp.join(root, "exp_ultimate_freeSolv_4*RDrun_*"))
    ts_exps.sort()
    ft_aux_exps = glob(osp.join(root, "exp_ultimate_freeSolv_11_*_RDrun_*"))
    ft_aux_exps.append(osp.join(root, "exp_ultimate_freeSolv_14_RDrun_2022-05-20_100955__472979"))
    ft_aux_exps.sort()
    plt.figure(figsize=(5, 4))
    for name, exps in [("Train-from-scratch", ts_exps), ("Fine-tune", ft_exps)]:
                       # ("Fine-tune-combined", ft_aux_exps)]:
        train_sizes = []
        rmse_mean = []
        rmse_std = []
        for exp in exps:
            df_all = pd.read_csv(osp.join(exp, "df_all.csv"))
            if name == "Fine-tune-combined":
                rmse = df_all["test_RMSE_CalcSol"]
                train_sizes.append(df_all.iloc[0]["train_n_units_MSE_CalcSol"])
            else:
                rmse = df_all["test_RMSE_activity"]
                train_sizes.append(df_all.iloc[0]["train_n_units"])
            rmse_mean.append(rmse.mean())
            rmse_std.append(rmse.std())
        plt.errorbar(train_sizes, rmse_mean, yerr=rmse_std, label=name, fmt="-o", capsize=3)
    plt.xlabel("FreeSolv training size")
    plt.ylabel("Test RMSE, kcal/mol")
    plt.title("Learning curve on FreeSolv-MMFF")
    plt.legend()
    plt.savefig(osp.join(root, "learning_curve.png"))


if __name__ == '__main__':
    learning_curve()
