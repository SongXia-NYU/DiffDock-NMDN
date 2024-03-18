import ase
from typing import List
from argparse import Namespace
from torch_geometric.data import Batch
from Networks.kano.kano4mdn import KanoAtomEmbed
from chemprop_kano.features.featurization import MolGraph
from utils.data.data_utils import get_ion_z

atom_num2_smiles = {}
for atom_num in range(95):
    atom_num2_smiles[atom_num] = f"[{ase.Atom(atom_num).symbol}]"


class Kano4Metal(KanoAtomEmbed):
    """
    KANO model for metal ions.
    """
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

    def get_mol_graphs(self, runtime_vars: dict) -> List[MolGraph]:
        data_batch: Batch = runtime_vars["data_batch"]
        smiles_list = [atom_num2_smiles[atom_num.item()] for atom_num in get_ion_z(data_batch).view(-1)]

        mol_g_arg = Namespace(atom_messages=False)
        mol_graph_list = [MolGraph(smi, None, mol_g_arg, False) for smi in smiles_list]
        return mol_graph_list

    def load_kano_ds(self, cfg: dict):
        pass
