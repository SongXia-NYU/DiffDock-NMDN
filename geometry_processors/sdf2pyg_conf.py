import argparse
import os

import torch

from geometry_processors.DataPrepareUtils import my_pre_transform, set_force_cpu


def sdf2pyg_single(conf_sdf, out_path, add_mol_id):
    from rdkit.Chem import SDMolSupplier
    with SDMolSupplier(conf_sdf) as suppl:
        mol_list = [mol for mol in suppl]

    data_list = []
    for mol in mol_list:
        coordinate = mol.GetConformer().GetPositions()
        coordinate = torch.as_tensor(coordinate)
        elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        elements = torch.as_tensor(elements).long()
        info_dict = {
            "R": coordinate,
            "Z": elements,
            "N": torch.as_tensor([len(elements)]).long()
        }

        if add_mol_id:
            for name in ["cluster_no", "initial_conformation_id", "molID"]:
                info_dict[name] = mol.GetProp(name)

        from torch_geometric.data import Data
        _this_data = Data(**info_dict)

        _this_data = my_pre_transform(_this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                      cutoff=10.0, boundary_factor=100., use_center=True, mol=None,
                                      cal_3body_term=False, bond_atom_sep=False, record_long_range=True)

        data_list.append(_this_data)

    torch.save(data_list, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_sdf")
    parser.add_argument("--out_path")
    parser.add_argument("--add_mol_id", action="store_true", help="Processing Dongdong's NMR data only")
    args = parser.parse_args()
    args = vars(args)

    set_force_cpu()
    try:
        sdf2pyg_single(**args)
    except Exception as e:
        print(f"Error processing {args['conf_sdf']}: {e}")
        os.system(f"echo FAILED > {args['out_path']}")


if __name__ == '__main__':
    main()
