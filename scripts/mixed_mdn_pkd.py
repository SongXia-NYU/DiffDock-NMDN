import pandas as pd
import torch
import numpy as np
import glob
import os
import os.path as osp

from utils.scores.casf_scores import calc_screening_score

def mix_mdn_pkd(mdn_folder, pkd_folder, out_folder):
    save_screen_folder = osp.join(out_folder, "casf-scores", "screening-mixed")
    pred_data_folder = osp.join(save_screen_folder, "pred_data")
    os.makedirs(pred_data_folder, exist_ok=True)

    mdn_pred_folder = osp.join(mdn_folder, "casf-scores", "screening", "pred_data")
    pkd_pred_folder = osp.join(pkd_folder, "casf-scores", "screening", "pred_data")
    target_pdbs = [osp.basename(f).split("_score.dat")[0] for f in glob.glob(osp.join(mdn_pred_folder, "*_score.dat"))]
    target_pdbs = []

    for target_pdb in target_pdbs:
        mdn_df = pd.read_csv(osp.join(mdn_pred_folder, f"{target_pdb}_score.dat"), sep=" ")
        pkd_df = pd.read_csv(osp.join(pkd_pred_folder, f"{target_pdb}_score.dat"), sep=" ")

        pkd_df["binder_pdb"] = pkd_df["#code_ligand_num"].map(lambda s: s.split("_")[0])
        binder_pdbs = set(pkd_df["binder_pdb"].values.tolist())
        the_chosen_ones = []
        for binder_pdb in binder_pdbs:
            mdn_prob_clone = mdn_df["score"].values.copy()
            mdn_prob_clone[pkd_df["binder_pdb"] != binder_pdb] = -999999
            the_chosen_ones.append(np.argmax(mdn_prob_clone))
        decoy_masker = torch.zeros(pkd_df.shape[0]).bool().fill_(True)
        decoy_masker[the_chosen_ones] = False
        pkd_df.loc[decoy_masker.numpy(), "score"] = -999999

        pkd_df = pkd_df[["#code_ligand_num", "score"]]
        pkd_df.to_csv(osp.join(pred_data_folder, f"{target_pdb}_score.dat"), sep=" ", index=False)
    
    result_summary = calc_screening_score(save_screen_folder, "pred_data", "screening")
    print(result_summary)

if __name__ == "__main__":
    mix_mdn_pkd("results/exp_pl_112_run_2022-11-09_132555__913386", "results/exp_pl_063_run_2022-10-11_211158__568415/", "exp_pl_123_run")
