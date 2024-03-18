import argparse

import pandas as pd
import os.path as osp


def read_xlogp3(input_list, out_csv):
    dfs = []
    for input_file in input_list:
        with open(input_file) as f:
            try:
                this_logp = float(f.read().split()[-1])
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                print(f"File content: {f.read()}")
        sample_id = osp.basename(input_file).split(".")[0]
        this_df = pd.DataFrame({"sample_id": [sample_id], "calcLogP": [this_logp]})
        dfs.append(this_df)

    out_df = pd.concat(dfs, axis=0)
    out_df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list")
    parser.add_argument("--out_csv")
    args = parser.parse_args()
    args = vars(args)

    with open(args["input_list"]) as f:
        args["input_list"] = f.read().split()

    read_xlogp3(**args)


if __name__ == '__main__':
    main()
