import argparse
import time

import pandas as pd
import os.path as osp

import tqdm


def split_csv(csv, out_folder, use_tqdm):
    df = pd.read_csv(csv, dtype={"SourceFile": str})
    ids = list(range(df.shape[0]))
    if use_tqdm:
        ids = tqdm.tqdm(ids)
    for i in ids:
        this_df = df.iloc[[i]]
        file_handle = this_df.iloc[0]["FileHandle"]
        this_df.to_csv(osp.join(out_folder, f"{file_handle}.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv")
    parser.add_argument("--out_folder")
    parser.add_argument("--use_tqdm", action="store_true")
    args = parser.parse_args()
    args = vars(args)

    split_csv(**args)


if __name__ == '__main__':
    main()
