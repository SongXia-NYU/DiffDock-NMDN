import argparse
import os
import os.path as osp
from typing import List
import numpy as np
import pandas as pd
from ase.units import Hartree, eV


hartree2ev = Hartree / eV
ev2kcal = 23.06035
# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15


def calc_sol(args):
    sol_water = "water"
    sol_gas = "gas"
    sol_octanol = "1-octanol"
    try:
        df_water = pd.read_csv(args["water"], dtype={"sample_id": np.int32}).set_index("sample_id").add_prefix(sol_water+"_")
    except ValueError:
        df_water = pd.read_csv(args["water"]).set_index("sample_id").add_prefix(sol_water+"_")
    try:
        df_gas = pd.read_csv(args["gas"], dtype={"sample_id": np.int32}).set_index("sample_id").add_prefix(sol_gas+"_")
    except ValueError:
        df_gas = pd.read_csv(args["gas"]).set_index("sample_id").add_prefix(sol_gas+"_")
    try:
        df_octanol = pd.read_csv(args["octanol"], dtype={"sample_id": np.int32}).set_index("sample_id").add_prefix(sol_octanol + "_")
    except ValueError:
        df_octanol = pd.read_csv(args["octanol"]).set_index("sample_id").add_prefix(sol_octanol + "_")
    result = df_water.join(df_gas)
    result = result.join(df_octanol)

    source = args["source"]
    if source in ["xtb", "orca"]:
        if df_water.shape != (0, 0) and df_gas.shape != (0, 0):
            diff = result[f"{sol_water}_total_energy(Eh)"] - result[f"{sol_gas}_total_energy(Eh)"]
            result[f"{sol_water}_{sol_gas}(kcal/mol)"] = diff * hartree2ev * ev2kcal
        if df_water.shape != (0, 0) and df_octanol.shape != (0, 0):
            diff1 = result[f"{sol_water}_total_energy(Eh)"] - result[f"{sol_octanol}_total_energy(Eh)"]
            result[f"{sol_water}_{sol_octanol}(kcal/mol)"] = diff1 * hartree2ev * ev2kcal
            result["calcLogP"] = result[f"{sol_water}_{sol_octanol}(kcal/mol)"] / logP_to_watOct
    elif source in ["gauss", "gaussian"]:
        gas = result[f"{sol_gas}_E(eV)"]
        water = result[f"{sol_water}_E(eV)"]
        octanol = result[f"{sol_octanol}_E(eV)"]
        result[f"{sol_water}_{sol_gas}(kcal/mol)"] = (water - gas) * ev2kcal
        result[f"{sol_octanol}_{sol_gas}(kcal/mol)"] = (octanol - gas) * ev2kcal
        result[f"{sol_water}_{sol_octanol}(kcal/mol)"] = (water - octanol) * ev2kcal
        result[f"calcLogP"] = result[f"{sol_water}_{sol_octanol}(kcal/mol)"] / logP_to_watOct
    else:
        raise ValueError(f"Invalid source: {source}")

    result.to_csv(args["out_path"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--water", type=str, default=None)
    parser.add_argument("--gas", type=str, default=None)
    parser.add_argument("--octanol", type=str, default=None)
    parser.add_argument("--out_path", type=str, default="tmp.csv")
    parser.add_argument("--source", default="xtb")
    args = vars(parser.parse_args())
    calc_sol(args)

    
if __name__ == '__main__':
    main()
