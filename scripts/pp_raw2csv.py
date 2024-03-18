from typing import List
import pandas as pd
from glob import glob
import torch
import os.path as osp
import argparse


def raw2csv_amend():
    rundir = "exp_pp_007_run_2023-10-09_150628__057299"
    testdir = glob(osp.join(rundir, "exp_pp_???_test_on_pp_test_decoys2_separated_*"))[0]
    test_pt = osp.join(testdir, "loss_prot_prot_test_decoy2_separated.polar.polar_test.pt")
    record_csv = osp.join(testdir, "record_name_prot_prot_test_decoy2_separated.polar.polar.csv")

    amend_csv = "/scratch/sx801/scripts/Mol3DGenerator/scripts/ProtProt_Xuhang/prot_prot_test_decoy2_separated.polar.polar.amend.csv"
    amend_df = pd.read_csv(amend_csv).set_index("sample_id").drop("Unnamed: 0", axis=1)

    test_d = torch.load(test_pt, map_location="cpu")
    test_info = {"sample_id": test_d["sample_id"], "score_un-normalized": test_d["PROP_PRED"]}
    test_info["score_normalized"] = test_d["MDN_SUM_DIST2_REF9.0"]
    test_df = pd.DataFrame(test_info).set_index("sample_id")
    record_df = pd.read_csv(record_csv).set_index("sample_id")
    out_df = record_df.join(test_df)
    out_df = amend_df.join(out_df)
    exp_id = "_".join(rundir.split("_")[1:3])
    out_df.to_csv(osp.join(testdir, f"{exp_id}.csv"))

def raw2csv(rundir: str):
    testdirs: List[str] = glob(osp.join(rundir, "exp_pp_???_test_on_*"))
    for testdir in testdirs:
        try:
            test_pt = glob(osp.join(testdir, "loss_*_test.pt"))[0]
            record_csv = glob(osp.join(testdir, "record_name_*.csv"))[0]
        except IndexError:
            continue

        test_d = torch.load(test_pt, map_location="cpu")
        test_info = {"sample_id": test_d["sample_id"], "score_un-normalized": test_d["PROP_PRED"]}
        test_info["score_normalized"] = test_d['MDN_LOGSUM_DIST2_REFDIST2']
        test_df = pd.DataFrame(test_info).set_index("sample_id")
        record_df = pd.read_csv(record_csv).set_index("sample_id")
        out_df = record_df.join(test_df)
        exp_id = "_".join(rundir.split("_")[1:3])
        out_df.to_csv(osp.join(testdir, f"{exp_id}.af22_50.csv"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str)
    args = parser.parse_args()
    raw2csv(args.folder_name)

if __name__ == "__main__":
    main()
