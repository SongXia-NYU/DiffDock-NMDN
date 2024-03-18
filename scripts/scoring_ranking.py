import glob
import math

import numpy as np
import os.path as osp

import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

from utils.LossFn import pKd2deltaG
from utils.eval.TestedFolderReader import TestedFolderReader

"""
Plot Exp vs. Calculated on the CASF-2016 test set and calculate Pearson R
"""


def main():
    reader = TestedFolderReader("exp_pl_005_run_2022-06-03_164656__676997", "exp_pl_005_test_on_casf2016-scoring_2022-09-02_143746", root="..")
    # folder = "../exp_pl/exp_pl_003_run_2022-05-31_130330__075963/exp_pl_003_test_2022-05-31_183408"

    loss_d = reader.result_mapper["test"]
    exp = loss_d["PROP_TGT"].view(-1).numpy()
    cal = loss_d["PROP_PRED"].view(-1).numpy()

    r2 = plot_scatter_info(exp, cal, reader.save_root, "exp_vs_cal.png", "Experimental pKd vs. Calculated pKd")

    ss = pd.read_csv("/Users/songxia/Documents/PycharmProjects/Mol3DGenerator/data/PDBbind_v2020/csv/CASF-2016.csv")
    pdb = [s.split("_")[0] for s in reader.record_mapper["CASF-2016_pl_06022022"]["protein_file"].values]
    df = pd.DataFrame({"pdb": pdb, "PRED": cal.tolist(), "TGT": exp.tolist()}).set_index("pdb")
    df = pd.concat([df, ss.set_index("pdb")], axis=1)
    df = df.reset_index()
    spearman, kendall = get_rank(df, ss)

    rank_result = pd.DataFrame({"spearman": [spearman], "kendall": [kendall]})
    rank_result.to_csv(osp.join(reader.save_root, "rank_result.csv"), index=False)

    loss_d["score_r"] = math.sqrt(r2)
    loss_d["rank_spearman"] = spearman
    loss_d["rank_kendall"] = kendall
    torch.save(loss_d, reader.result_file_mapper["test"])


if __name__ == '__main__':
    main()
