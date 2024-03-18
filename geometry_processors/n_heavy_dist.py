import pandas as pd
from rdkit.Chem import MolFromSmiles
import matplotlib.pyplot as plt


def show_n_heavy_dist(smiles_list):
    n_heavy_list = []
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        n_heavy_list.append(mol.GetNumHeavyAtoms())

    plt.hist(n_heavy_list, bins=30)
    plt.savefig("freesolv_nheavy")


def main():
    df = pd.read_csv("../data/freesolv_sol/freesolv_paper.csv")
    smiles_list = df["cano_smiles"].values.tolist()
    show_n_heavy_dist(smiles_list)


if __name__ == '__main__':
    main()
