from typing import Union
import torch
from torch import device, dtype
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.batch import Batch

from utils.data.MyData import MyData
from utils.utils_functions import get_device

# deal with HeteroData
def infer_device(data: Union[Data, Batch, dict]) -> device:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.R.device
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].R.device

# infer floating point precision
def infer_type(data: Union[Data, Batch]) -> dtype:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.R.dtype
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].R.dtype

def get_prot_coords(data: Union[Data, Batch]) -> torch.Tensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): 
        return data["graph"].node_position

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.R_prot_pad
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["protein"].R

def get_prot_natom(data: Union[Data, Batch]) -> torch.Tensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict):
        from torchdrug.data.protein import PackedProtein
        prot: PackedProtein = data["graph"]
        return prot.num_nodes

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.N_prot
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["protein"].N

def get_lig_natom(data: Union[Data, Batch]) -> torch.Tensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.N
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].N

def get_lig_coords(data: Union[Data, Batch]) -> torch.Tensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.R
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].R

def get_lig_z(data: Union[Data, Batch]) -> torch.Tensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    # atomic number of each atom
    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.Z
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].Z

def get_pl_edge(data: Union[Data, Batch]) -> torch.LongTensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): 
        raise ValueError("PL edge will be computed on the fly")

    d0 = data.get_example(0) if isinstance(data, Batch) else data
    if isinstance(d0, MyData):
        return data.PL_min_dist_sep_oneway_edge_index
    
    assert isinstance(d0, HeteroData), data.__class__
    return data[("ligand", "interaction", "protein")].min_dist_edge_index

# ----------------------- Works for Batch ------------------------- #
def get_lig_batch(data: Union[HeteroData, Batch]) -> torch.LongTensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    if isinstance(data, HeteroData):
        return torch.as_tensor([0 for __ in range(data["ligand"].R.shape[0])]).to(get_device())

    d0 = data.get_example(0)
    if isinstance(d0, MyData):
        return data.atom_mol_batch
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].batch

def get_num_mols(data: Union[HeteroData, Batch]) -> torch.LongTensor:
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict): data = data["ligand"]

    # single batch, only used during prediction
    if isinstance(data, HeteroData):
        return 1

    d0 = data.get_example(0)
    if isinstance(d0, MyData):
        return data.N.shape[0]
    
    assert isinstance(d0, HeteroData), data.__class__
    return data["ligand"].N.shape[0]

def data_to_device(data: Union[Batch, Data, dict]):
    if isinstance(data, (Batch, Data)):
        return data.to(get_device())
    
    # PL dataset from ESM-GearNet
    for key in data.keys():
        data[key] = data[key].to(get_device())
    return data

def get_ion_z(data: Union[Batch, Data, dict]):
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict):
        data = data["ligand"]
    
    return data["ion"].Z

def get_sample_id(data: Union[Batch, Data, dict]):
    return get_prop(data, "sample_id")

def get_prop(data: Union[Batch, Data, dict], name: str):
    # data_batch is a dict when using ESM-GearNet
    if isinstance(data, dict):
        data = data["ligand"]

    return getattr(data, name)

