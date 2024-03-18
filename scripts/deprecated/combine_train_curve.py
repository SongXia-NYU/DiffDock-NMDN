import os.path as osp

import torch

from scripts.training_curve import print_training_curve
from utils.utils_functions import get_device


def combine_train_curve(folder1, folder2):
    loss_data1 = torch.load(osp.join(folder1, 'loss_data.pt'), map_location=get_device())
    loss_data2 = torch.load(osp.join(folder2, 'loss_data.pt'), map_location=get_device())
    epoch = loss_data1[-1]["epoch"]
    for loss in loss_data2:
        loss["epoch"] += epoch
        # loss["v_emae"] = loss["MAE_gasEnergy"]
        # loss["v_ermse"] = loss["RMSE_gasEnergy"]
        loss_data1.append(loss)
    print_training_curve(loss_data1, osp.join(folder2, "combined"))


if __name__ == '__main__':
    combine_train_curve("../../tmp/exp293_run_2021-06-16_134950",
                        "../../tmp/exp293-cont1_run_2021-06-21_154320")
