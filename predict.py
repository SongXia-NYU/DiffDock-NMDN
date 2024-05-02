from copy import deepcopy
from glob import glob
import os.path as osp

import torch
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from rdkit.Chem import AddHs
from rdkit.Chem.AllChem import SDMolSupplier

from geometry_processors.misc import ff_optimize
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.process.mp_pyg_runner import proc_hetero_graph
from utils.LossFn import post_calculation
from utils.data.DataPostProcessor import DataPostProcessor
from utils.data.MolFileDataset import SDFDataset
from utils.data.data_utils import get_num_mols
from utils.eval.predict import EnsPredictor
from utils.eval.tester import Tester
from utils.rmsd import symmetry_rmsd_from_mols

prot = "data/1e66_protein.pdb"
ligs = glob("data/1e66_1a30/rank*_confidence*.sdf")

tester = Tester("data/exp_pl_534_run_2024-01-22_211045__480688")
model = tester.model
model.eval()
collate_fn = Collater(None, None)
data_processor = DataPostProcessor(tester.cfg)
for i, lig_sdf in enumerate(ligs):
    data = proc_hetero_graph(MPInfo(protein_pdb=prot, ligand_sdf=lig_sdf))
    data = data_processor(data, 0)
    data = collate_fn([data])
    pred = model(data)
    breakpoint()

# atomic props
atomprop_predictor = EnsPredictor("./pretrained/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*")
sdf_ds = SDFDataset(ligs)
dl = DataLoader(sdf_ds, batch_size=len(ligs))
allh_batch = next(iter(dl))
model_pred = atomprop_predictor.predict_nograd(allh_batch)
phys_atom_prop = model_pred["atom_prop"]
phys_mol_prop: torch.Tensor = scatter_add(phys_atom_prop, allh_batch.atom_mol_batch, dim=0, dim_size=get_num_mols(data_batch))
phys_mol_prop = post_calculation(phys_mol_prop)

# RMSD info
rmsd_info = {}
for lig_sdf in ligs:
    splits = osp.basename(lig_sdf).split(".")
    target = splits[0]
    lig_name = ".".join(splits[1: -2])
    try:
        mol_supp = SDMolSupplier(lig_sdf, removeHs=True, sanitize=True, strictParsing=True)
    except OSError:
        continue
    mol = AddHs(mol_supp[0], addCoords=True)
    dst_mol = deepcopy(mol)
    ff_optimize(dst_mol, [0])
    rmsd = symmetry_rmsd_from_mols(mol, dst_mol, 1)
    rmsd_info[lig_sdf] = rmsd


