import argparse
import os

import numpy as np
import pandas as pd
from rdkit.Chem.AllChem import MolFromSmiles, MolToInchiKey, RemoveStereochemistry, RemoveAllHs, CanonicalizeMol

from geometry_processors.calc_sol import logP_to_watOct, ev2kcal


def get_intersection(ds1, ds2, csv1, csv2, csv_qm, out_csv):
    df1 = pd.read_csv(csv1, dtype={"sample_id": int})
    df_qm = pd.read_csv(csv_qm, dtype={"sample_id": int}).set_index("sample_id")
    df1["sample_id"] = np.arange(df1.shape[0])
    df2 = pd.read_csv(csv2, dtype={"FileHandle": int}).dropna().rename(
        {"Kow": "activity", "FileHandle": "sample_id"}, axis=1)

    def smiles2inchi(smiles):
        mol = MolFromSmiles(smiles)
        RemoveStereochemistry(mol)
        RemoveAllHs(mol)
        CanonicalizeMol(mol)
        return MolToInchiKey(mol)

    df1["source"] = [ds1] * df1.shape[0]
    df2["source"] = [ds2] * df2.shape[0]

    df1["inchi"] = df1["cano_smiles"].map(smiles2inchi)
    df2["inchi"] = df2["Smiles"].map(smiles2inchi)

    inchi1 = set(df1["inchi"].values.tolist())
    inchi2 = set(df2["inchi"].values.tolist())

    intersect_inchi = inchi1.intersection(inchi2)
    print(len(intersect_inchi))

    intersect_df = df1.merge(df2, on="inchi", how="inner", suffixes=("_1", "_2"))
    # intersect_df = intersect_df.sort_values(by="inchi")
    intersect_df = intersect_df.drop_duplicates(subset="inchi")
    intersect_df["gasEnergy"] = df_qm.loc[intersect_df["sample_id_2"].to_numpy()]["gas_E_atom(eV)"].values
    intersect_df = intersect_df.dropna()

    # post calculation
    intersect_df["CalcSol"] = intersect_df["activity_1"]
    intersect_df["CalcLogP"] = intersect_df["activity_2"]
    intersect_df["watOct"] = intersect_df["CalcLogP"] * logP_to_watOct
    intersect_df["watEnergy"] = intersect_df["gasEnergy"] + intersect_df["CalcSol"] / ev2kcal
    intersect_df["octEnergy"] = intersect_df["watEnergy"] - intersect_df["watOct"] / ev2kcal
    intersect_df["CalcOct"] = (intersect_df["octEnergy"] - intersect_df["gasEnergy"]) * ev2kcal

    os.makedirs("../../data/intersect_dataset", exist_ok=True)
    intersect_df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds1")
    parser.add_argument("--ds2")
    parser.add_argument("--csv1")
    parser.add_argument("--csv2")
    parser.add_argument("--csv_qm")
    parser.add_argument("--out_csv")
    args = parser.parse_args()
    args = vars(args)

    get_intersection(**args)


if __name__ == '__main__':
    main()

