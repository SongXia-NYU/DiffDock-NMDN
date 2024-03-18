import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def casf_custom_select():
    performance_csv = "exp_pl_422_run_2023-08-26_005727__219228/casf-custom-screen-scores-tune-mdn/performance_summay.csv"
    performance_df = pd.read_csv(performance_csv)
    def _get_mdn_sel(sel_str: str):
        if sel_str == "mdn_selAll": return np.nan
        return int(sel_str.split("mdn_sel")[-1])
    performance_df["mdn_sel_int"] = performance_df["mdn_sel"].map(_get_mdn_sel)
    performance_df = performance_df.dropna().sort_values(by="mdn_sel_int")

    topks = set(performance_df["topk"].values.tolist())
    for topk in topks:
        # if topk != "top20": continue
        this_df = performance_df[performance_df["topk"] == topk]
        plt.plot(this_df["mdn_sel_int"].values, this_df["EF"].values, "o-", label=topk)
    plt.xlabel("#Ligands selected by NMDN")
    plt.ylabel("TopK Enhancement Factor")
    plt.legend()
    plt.title("NMDN-pKd Selection Strategy")
    plt.tight_layout()
    plt.savefig("exp_pl_422_run_2023-08-26_005727__219228/casf-custom-screen-scores-tune-mdn/performance_summay.png")


def casf_topk_curve():
    performance_csv = "./exp_pl_422_run_2023-08-26_005727__219228/casf-custom-screen-scores/performance_summay.csv"
    performance_df = pd.read_csv(performance_csv)
    performance_df["topk"] = performance_df["topk"].map(lambda s: int(s.split("top")[-1]))
    performance_df = performance_df.rename(
        {"EF": "Enhancement Factor", "#TruePos": "Number of True Postives"}, axis=1).set_index("topk")

    mdn_df = performance_df[performance_df["mdn_sel"] == "mdn_selAll"]
    sns.lineplot(mdn_df[["Enhancement Factor", "Number of True Postives"]])
    plt.title("CASF-2016 Performance vs. TopK")
    plt.tight_layout()
    plt.savefig("exp_pl_422_run_2023-08-26_005727__219228/casf-custom-screen-scores/ef_vs_topk.png")


if __name__ == "__main__":
    casf_topk_curve()
