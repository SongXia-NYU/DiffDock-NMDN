import argparse
import numpy as np

import rdkit
from rdkit.Chem import SDMolSupplier, SDWriter


def write_fake_out(f_out):
    with open(f_out, "w") as f:
        f.write("FAILED!")


def min_sdf(f_confs, f_out):
    try:
        suppl = SDMolSupplier(f_confs, removeHs=False)
        lowest_e = np.inf
        selected_mol = None
        for mol in suppl:
            energy = float(mol.GetProp("energy_abs"))
            if energy < lowest_e:
                lowest_e = energy
                selected_mol = mol
        if selected_mol is not None:
            with SDWriter(f_out) as writer:
                writer.write(selected_mol)
        else:
            write_fake_out(f_out)
    except Exception as e:
        print(f"Something is wrong when processing {f_confs}: {e}")
        write_fake_out(f_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_confs")
    parser.add_argument("--f_out")
    args = parser.parse_args()
    args = vars(args)

    min_sdf(**args)


if __name__ == '__main__':
    main()
