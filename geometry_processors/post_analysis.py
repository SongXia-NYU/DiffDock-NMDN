import os
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
import numpy as np


def plot_scatter_info(exp_sol, cal_sol, save_folder, save_name, title, total_time=None, original_size=None):
    mae = np.mean(np.abs(exp_sol - cal_sol))
    rmse = np.sqrt(np.mean((exp_sol - cal_sol)**2))
    r = pearsonr(exp_sol, cal_sol)[0]
    r2 = r ** 2

    plt.figure()
    xy = np.vstack([exp_sol, cal_sol])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    plt.scatter(exp_sol[idx], cal_sol[idx], c=z[idx])
    x_min = min(exp_sol)
    x_max = max(exp_sol)
    plt.plot([x_min, x_max], [x_min, x_max], color="red", label="y==x")
    plt.xlabel("Experimental")
    plt.ylabel(f"Calculated")
    plt.legend()
    plt.title(title)
    annotate = f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR2 = {r2:.4f}\n"
    if total_time is not None:
        annotate = annotate + f"Total Time = {total_time:.0f} seconds\n"
    if original_size is not None:
        annotate = annotate + f"Showing {len(exp_sol)} / {original_size}\n"
    plt.annotate(annotate, xy=(0.05, 0.70), xycoords='axes fraction')
    plt.savefig(osp.join(save_folder, save_name))
    plt.close()
    return r2


def post_analysis():
    """
    Only works if all calculation successes.
    :return:
    """
    from rdkit.Chem import MolFromSmiles, MolToInchi, RemoveHs, RemoveStereochemistry

    # LEVELS = ["crude", "sloppy", "loose", "lax", "normal", "tight", "vtight", "extreme"]
    LEVELS = ["crude"]
    root = "../data/lipop_sol"
    gas = "sp"
    water = "sp"
    # gauss = "gauss_"
    gauss = "orca_"
    # title = "XTB calculated at {level} Level vs. Experimental"
    title = "XTB water opt {level} level, ORCA single point vs. Experimental"

    opt_process = f"gas{gas}_water{water}"
    save_folder = osp.join(root, "post_analysis", "orca_logP")
    os.makedirs(save_folder, exist_ok=True)
    exp_df = pd.read_csv(osp.join(root, "lipop_paper.csv"))
    # exp_sol = exp_df["activity"].values
    exp_smiles = exp_df["cano_smiles"].values.tolist()

    r2s = []
    wall_times = []

    for level in LEVELS:
        p_csv = osp.join(root, "summary", "orca_logP", "summary_spsp_sol_crude.csv")
        cal_df = pd.read_csv(p_csv, dtype={"sample_id": np.int32}).sort_values(by="sample_id").set_index("sample_id").dropna()
        cal = cal_df[f"calcLogP"].values
        cal_smiles_series = cal_df["calcLogP"]
        exp = exp_df["activity"].values[cal_df.index]
        for sample_id in cal_df.index:
            smiles1 = exp_smiles[sample_id]
            smiles2 = cal_smiles_series.loc[sample_id]
            if isinstance(smiles2, str):
                # assert smiles1 == smiles2
                mol1 = RemoveHs(MolFromSmiles(smiles1))
                mol2 = RemoveHs(MolFromSmiles(smiles2))
                RemoveStereochemistry(mol1)
                RemoveStereochemistry(mol2)
                assert MolToInchi(mol1) == MolToInchi(mol2)

        total_time = np.sum(cal_df[f"octanol_wall_time(secs)"].values + cal_df[f"water_wall_time(secs)"].values)
        r2 = plot_scatter_info(exp, cal, save_folder, total_time=total_time, title=title.format(level=level),
                               save_name=f"exp_vs_{level}.png", original_size=exp_df.shape[0])
        r2s.append(r2)
        wall_times.append(total_time)

    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(LEVELS, r2s, label="R2", marker="o")
    ax1.set_xlabel("XTB Level")
    ax1.set_ylabel("R2")
    ax1.set_ylim([0, 1])
    ax1.set_title("XTB calculated solvation energy R2 and total time")
    ax2 = ax1.twinx()
    ax2.plot(LEVELS, wall_times, label="total time (s)", color="red", marker="o")
    ax2.set_ylabel("Total Time (s)", color="red")
    plt.savefig(osp.join(save_folder, "exp_vs_all"))
    plt.close()


if __name__ == '__main__':
    post_analysis()
