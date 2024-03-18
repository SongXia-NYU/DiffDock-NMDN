import argparse

import pandas as pd
from DataGen.genconfs import runGenerator


def conf_gen(name, smiles, source, save_root, **extra):
    runGenerator([name], [smiles], source, save_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--csv")
    parser.add_argument("--source")
    parser.add_argument("--save_root")
    args = parser.parse_args()
    args = vars(args)

    info_csv = pd.read_csv(args["csv"], dtype={"FileHandle": str})[["FileHandle", "Smiles"]].set_index("FileHandle")
    smiles = info_csv.loc[args["name"]].values.item()
    args["smiles"] = smiles

    conf_gen(**args)


if __name__ == '__main__':
    main()
