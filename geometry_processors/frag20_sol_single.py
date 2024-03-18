import argparse
import os
import os.path as osp
import time

import pandas as pd
import torch
from ase.units import Hartree, eV
from torch_geometric.data import Data

from geometry_processors import gauss
import geometry_processors.gauss.read_gauss_log
from geometry_processors.DataPrepareUtils import my_pre_transform
from geometry_processors.sdf2pyg_single import sdf2pyg_single

hartree2ev = Hartree / eV


def frag20_csd20_sol_single(sample_id, csv, geometry, out_path, df=None):
    # t0 = time.time()
    if df is None:
        df = pd.read_csv(csv)
    this_sdf = sdf_from_sample_id(sample_id, geometry)

    # print(f"init: {time.time()-t0}")
    sdf2pyg_single(this_sdf, df, out_path)


def sdf_from_sample_id(sample_id, geometry):
    source = sample_id.split("_")[0]
    idx = sample_id.split("_")[1]

    if geometry == "qm":
        extra = ".opt"
    else:
        extra = ""

    try:
        source = int(source)
        this_sdf = f"/ext3/Frag20_{source}_data/{idx}{extra}.sdf"
    except ValueError:
        assert source in ["PubChem", "Zinc", "CCDC"]
        if source in ["PubChem", "Zinc"]:
            this_sdf = f"/ext3/Frag20_9_data/{source.lower()}/{idx}{extra}.sdf"
        else:
            if geometry == "qm":
                extra = ".opt"
            else:
                extra = "_min"
            this_sdf = f"/ext3/CSD20_sol/cry_min/{idx}{extra}.sdf"
    return this_sdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="20_11110")
    parser.add_argument("--csv", default="20_11110.csv")
    parser.add_argument("--geometry", default="qm")
    parser.add_argument("--out_path", default="20_11110.pyg")
    args = parser.parse_args()
    args = vars(args)

    frag20_csd20_sol_single(**args)


if __name__ == '__main__':
    main()

