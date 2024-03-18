import os.path as osp
import os
from math import log10

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.utils_functions import get_device


def print_training_curve(loss_csv, save_path):
    # training curve
    os.makedirs(save_path, exist_ok=True)
    t_loss = loss_csv["train_loss"].values
    # pad training loss before training
    t_loss[0] = t_loss[1]
    v_loss = loss_csv["valid_loss"].values

    interested_properties = ["MAE_gasEnergy", "MSE_gasEnergy", "MAE_E", "MSE_E", "v_emae", "v_emse",
                             "MAE_CalcSol", "MAE_CalcOct", "MAE_y", "MSE_y", "MAE_watOct", "MAE_activity",
                             "RMSE_activity"]
    interested_array = {}
    for p in interested_properties:
        display_name = p
        change = 1.
        if p in ["v_emae", "MAE_E"]:
            display_name = "MAE_E kcal/mol"
            change = 23.061
        elif p in ["v_emse", "MSE_E"]:
            display_name = "MSE_E kcal/mol"
            change = 23.061
        elif p in ["MAE_gasEnergy", "MSE_gasEnergy"]:
            display_name = p + " kcal/mol"
            change = 23.061
        elif p == "MAE_CalcSol":
            display_name = "MAE water sol. kcal/mol"
        elif p == "MAE_CalcOct":
            display_name = "MAE oct sol. kcal/mol"
        elif p in ["MAE_y", "MSE_y"]:
            display_name = "{} NMR".format(p.split('_')[0])
        if p in loss_csv.columns:
            interested_array[display_name] = loss_csv[p].values * change
    tmp = [t_loss, v_loss]
    for p in interested_array:
        tmp.append(interested_array[p])
    concat = np.concatenate(tmp)
    total_min = np.min(concat)
    total_max = np.max(concat)

    plt.figure(figsize=(10, 8))
    # plt.plot(e_mae_v, label='Energy MAE, valid')
    plt.plot(t_loss, label='Loss, train')
    plt.plot(v_loss, label='Loss, valid')
    for p in interested_array:
        plt.plot(interested_array[p], label=p)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training_curve')
    plt.savefig(osp.join(save_path, 'training_curve'))
    for num, diff in enumerate(torch.logspace(-2, log10(total_max-total_min), steps=10)):
        plt.ylim([total_min, diff+total_min])
        plt.savefig(osp.join(save_path, 'training_curve_{}'.format(num + 1)))
    plt.close()


def training_curve_folders(name):
    from glob import glob
    folders = glob(name)
    for folder in folders:
        loss_data = pd.read_csv(osp.join(folder, "loss_data.csv"))
        print_training_curve(loss_data, folder)


if __name__ == '__main__':
    training_curve_folders("../exp_pl_005_run_2022-06-03_164656__676997")
    # training_curve_folders("../../raw_data/qm9-nmr/*exp1_*")
