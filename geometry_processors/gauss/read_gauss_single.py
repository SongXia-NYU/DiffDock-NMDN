import argparse

import pandas as pd
import torch

from geometry_processors.gauss.read_gauss_log import Gauss16Log


def read_gauss_single():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--log_sdf", type=str)
    parser.add_argument("--gauss_version", type=int, default=16)
    parser.add_argument("--out_pyg", default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    gauss_log = Gauss16Log(args.log, args.log_sdf, gauss_version=args.gauss_version)

    if args.out_csv is not None:
        result = gauss_log.get_data_frame()
        result.to_csv(args.out_csv, index=False)

    if args.out_pyg is not None:
        torch.save(gauss_log.get_torch_data(add_edge=True), args.out_pyg)


if __name__ == '__main__':
    read_gauss_single()
