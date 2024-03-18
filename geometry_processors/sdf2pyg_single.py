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
from geometry_processors.DataPrepareUtils import my_pre_transform, set_force_cpu

hartree2ev = Hartree / eV


def sdf2pyg_single(this_sdf, df, out_path):
    # t0 = time.time()

    sdf_info = gauss.read_gauss_log.Gauss16Log(log_path=None, log_sdf=this_sdf, supress_warning=True)

    # print(f"init gauss info: {time.time()-t0}")
    # t0 = time.time()

    if df is not None:
        this_info: dict = df.iloc[0].to_dict()
        if "gasEnergy" not in this_info.keys():
            if "gasEnergy(au)" in this_info.keys():
                for key in ["gasEnergy", "watEnergy", "octEnergy"]:
                    # convert unit and subtract reference energy
                    this_info[key] = hartree2ev * this_info[key + "(au)"] - sdf_info.reference_u0
            else:
                this_info["gasEnergy"] = this_info.pop("gas_E_atom(eV)")
                this_info["watEnergy"] = this_info.pop("water_E_atom(eV)")
                this_info["octEnergy"] = this_info.pop("1-octanol_E_atom(eV)")
                this_info["CalcSol"] = this_info.pop("water_gas(kcal/mol)")
                this_info["CalcOct"] = this_info.pop("1-octanol_gas(kcal/mol)")
                this_info["watOct"] = this_info.pop("water_1-octanol(kcal/mol)")
        gauss.read_gauss_log.Gauss16Log.conv_type(this_info)
        this_info.update(sdf_info.get_basic_dict())
    else:
        this_info = sdf_info.get_basic_dict()

    # print(f"init dict: {time.time()-t0}")
    # t0 = time.time()

    this_data = Data(**this_info)

    # print(f"init data: {time.time()-t0}")
    # t0 = time.time()

    this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                 cutoff=10.0, boundary_factor=100., use_center=True, mol=None, cal_3body_term=False,
                                 bond_atom_sep=False, record_long_range=True)

    # print(f"edge cal: {time.time()-t0}")
    # t0 = time.time()
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    torch.save(this_data, out_path)

    # print(f"save: {time.time()-t0}")
    # t0 = time.time()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--this_sdf")
    parser.add_argument("--out_path")
    args = parser.parse_args()
    args = vars(args)

    set_force_cpu()
    try:
        sdf2pyg_single(**args, df=None)
    except Exception as e:
        print(f"Error processing {args['this_sdf']}: {e}")
        os.system(f"echo FAILED > {args['out_path']}")


if __name__ == '__main__':
    main()
