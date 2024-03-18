from typing import Callable, List, Optional
from torch_geometric.data import Dataset, InMemoryDataset, Data, HeteroData
from utils.DataPrepareUtils import bn_edge_calculate_fast, pairwise_dist
import torch
import os.path as osp
from rdkit.Chem.AllChem import SDMolSupplier
from rdkit.Chem import AddHs
from tqdm import tqdm

from utils.data.MyData import MyData
from utils.utils_functions import lazy_property, get_device

class MolFileDataset(Dataset):
    """
    Construct TorchGeometric data set directly from mole file format (sdf, mol2, etc..)
    """
    def __init__(self, mol_files: List[str]) -> None:
        super().__init__()
        self.mol_files = mol_files

    def get(self, idx: int):
        mol_file = self.mol_files[idx]
        reader = self.get_data_reader(mol_file)
        ds_dict = reader.get_basic_dict()
        ds_dict = bn_edge_calculate_fast(ds_dict)
        ds_dict["mol_file"] = [mol_file]
        d = MyData(**ds_dict)
        return d

    def get_data_reader(self, mol_file):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.mol_files)

    def len(self) -> int:
        return len(self)


class SDFDataset(MolFileDataset):
    def __init__(self, sdf_files) -> None:
        super().__init__(sdf_files)

    def get_data_reader(self, mol_file):
        from utils.ConfReader import SDFReader
        return SDFReader(mol_file)
    
class StackedSDFileDataset(Dataset):
    """
    Construct pyg data from SDF files, each sdf file may contain multiple molecules
    """
    def __init__(self, sdf_files: List[str]) -> None:
        super().__init__()
        self.sdf_files = sdf_files

    def get(self, idx: int):
        reader = self.mol_readers[idx]
        ds_dict = reader.get_basic_dict()
        ds_dict["R"] = ds_dict["R"].to(get_device())
        ds_dict = bn_edge_calculate_fast(ds_dict)
        ds_dict["file_handle_ranked"] = reader.file_handle_ranked
        d = MyData(**ds_dict)
        return d
    
    @lazy_property
    def mol_readers(self):
        from utils.ConfReader import MolReader
        readers = []
        for sdf_file in tqdm(self.sdf_files, desc="Read Mol Files"):
            try:
                supp = SDMolSupplier(sdf_file, removeHs=True)
            except OSError:
                continue
            for mol in supp:
                if mol is None: continue
                mol = AddHs(mol, addCoords=True)
                if mol is None: continue
                reader = MolReader(mol)
                reader.file_handle_ranked = self.retrieve_fl(mol, sdf_file)
                readers.append(reader)
        return readers
    
    def retrieve_fl(self, mol, sdf_file: str):
        if "/fep-benchmark/pose_diffdock/polarh/" in sdf_file:
            fl = osp.basename(sdf_file).split(".polarh.sdf")[0]
            rank_name: str = mol.GetProp("_Name")
            rank: str = rank_name.split("_confidence")[0]
            return f"{fl}.{rank}"
        if "/fep-benchmark/" in sdf_file:
            target: str = sdf_file.split("/")[-3]
            lig_name: str = mol.GetProp("_Name")
            return f"{target}.{lig_name}"
        file_handle: str = osp.basename(osp.dirname(sdf_file))
        rank: str = osp.basename(sdf_file).split("_")[0]
        return f"{file_handle}.{rank}"

    def __len__(self) -> int:
        return len(self.mol_readers)

    def len(self) -> int:
        return len(self)

class MolFromPLDataset(InMemoryDataset):
    """
    A dataset that extract ligand from the Protein-Ligand structure. 
    It is currently only used to calculate the predicted logP by sPhysNet-MT on the PDBBind ligands.
    """
    def __init__(self, data_root, dataset_name):
        self.dataset_name = dataset_name
        super().__init__(data_root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx: int):
        pl_data = super().get(idx)
        if hasattr(pl_data, "confidence"):
            pl_data.BN_edge_index = pl_data.LIGAND_edge_index
            pl_data.num_BN_edge_index = torch.as_tensor([pl_data.LIGAND_edge_index.shape[1]])
            pl_data.atom_mol_batch = torch.zeros_like(pl_data.Z)
            file_handle: str = pl_data.file_handle
            rank: int = pl_data.rank.item()
            pl_data.file_handle_ranked = f"{file_handle}.rank{rank}"
            return MyData.from_data(pl_data)
        lig_dict = self.get_lig_dict(pl_data)
        if hasattr(pl_data, "ligand_file"):
            lig_dict["ligand_file"] = pl_data.ligand_file
        lig_dict = bn_edge_calculate_fast(lig_dict)
        d = MyData(**lig_dict)
        return d
    
    def get_lig_dict(self, pl_data):
        if isinstance(pl_data, HeteroData):
            lig_data = pl_data["ligand"]
            return {"R": lig_data.R, "Z": lig_data.Z, "N": lig_data.N, "protein_file": pl_data.protein_file}
        if not hasattr(pl_data, "N_l"):
            N_l = pl_data.N
        else:
            N_l = pl_data.N_l
        lig_dict = {"R": pl_data.R[:N_l, :], "Z": pl_data.Z[:N_l], "N": N_l, "protein_file": pl_data.protein_file}
        return lig_dict

    @property
    def raw_file_names(self):
        return ["dummy"]

    @property
    def processed_file_names(self):
        return [self.dataset_name]
