import matplotlib.pyplot as plt
import torch
import os.path as osp

from src.active_learning_vis import ALFolder


def learning_curve(folders: list, root="."):
    target = "MAE_CalcSol_each_avg"
    folders.sort(key=get_size)
    folders = [osp.join(root, f) for f in folders]
    training_sizes = []
    valid_mae = []
    test_mae = []
    plt.figure(figsize=(10, 8))
    for folder in folders:
        # extract training sizes from folder names
        training_size = get_size(folder)
        training_sizes.append(training_size)

        info = ALFolder(folder)
        info_df = info.info_df
        valid_mae.append(info_df.iloc[0][f"val_{target}"])
        test_mae.append(info_df.iloc[0][f"test_{target}"])

    plt.scatter(training_sizes, test_mae, marker='o')
    plt.title("Learning Curve on Frag20-Sol")
    plt.xlabel("Training size")
    plt.ylabel("Test Error, kcal/mol")
    for x, y in zip(training_sizes, test_mae):
        plt.annotate("{:.2f}".format(y), (x, y))
    # plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(osp.join(root, "learning_curve.png"))
    plt.show()
    plt.close()


def get_size(name):
    training_size = osp.basename(name).split("-")[0].split("exp")[-1][:-6]
    training_size = 1000 * int(training_size)
    return training_size


if __name__ == '__main__':
    _folders = [
        "exp21kinall-frag20Sol_active_ALL_2021-12-03_223745",
        "exp40kinall-frag20Sol_active_ALL_2021-12-04_212843",
        "exp80kinall-frag20Sol_active_ALL_2021-12-04_222155",
        "exp160kinall-frag20Sol_active_ALL_2021-12-04_222155",
        "exp320kinall-frag20Sol_active_ALL_2021-12-05_112531",
        "exp482kinall-frag20Sol_active_ALL_2021-12-08_133939"]
    learning_curve(_folders, "../../raw_data/exp_learning_curve_frag20_sol")
