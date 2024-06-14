from collections import defaultdict
from copy import deepcopy
from glob import glob
import os.path as osp
from typing import List, Optional
import pandas as pd
from tqdm import tqdm
import argparse

import torch
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from rdkit.Chem import AddHs
from rdkit.Chem.AllChem import SDMolSupplier

from geometry_processors.lm.esm_embedding import ESMCalculator
from geometry_processors.misc import ff_optimize
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.prot_utils import pdb2seq
from geometry_processors.process.mp_pyg_runner import proc_hetero_graph
from utils.LossFn import MDNMixLossFn, post_calculation
from utils.data.DataPostProcessor import DataPostProcessor
from utils.data.MolFileDataset import SDFDataset
from utils.data.data_utils import get_num_mols
from utils.eval.predict import EnsPredictor
from utils.eval.tester import Tester
from utils.rmsd import symmetry_rmsd_from_mols

parser = argparse.ArgumentParser()
parser.add_argument("--prot", type=str)
parser.add_argument("--ligs", type=str, nargs="+")
parser.add_argument("--nmdn_only", action="store_true")
parser.add_argument("--save_csv", default=None)
args = parser.parse_args()
prot: str = args.prot
ligs: List[str] = args.ligs
nmdn_only: bool = args.nmdn_only
save_csv: Optional[str] = args.save_csv

# the solvation propterties and RMSD info are only needed for pkd scores
if not nmdn_only:
    # solvation properties
    print("Calculating solvation energetics...")
    atomprop_predictor = EnsPredictor("./data/exp_frag20sol_012_active_ALL_2022-05-01_112820/exp_*_cycle_-1_*")
    sdf_ds = SDFDataset(ligs)
    dl = DataLoader(sdf_ds, batch_size=len(ligs))
    allh_batch = next(iter(dl))
    model_pred = atomprop_predictor.predict_nograd(allh_batch)
    phys_atom_prop = model_pred["atom_prop"]
    phys_mol_prop: torch.Tensor = scatter_add(phys_atom_prop, allh_batch.atom_mol_batch, dim=0, dim_size=get_num_mols(allh_batch))
    phys_mol_prop = post_calculation(phys_mol_prop)

    # RMSD info
    print("Calculate ligand stability features...")
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

# ESM-2 embedding
print("Computing ESM-2 embedding...")
esm_calculator = ESMCalculator(None)
seq = pdb2seq(prot)
prot_embed = esm_calculator.embed_from_seq(seq).squeeze(0)[1: -1, :].float()

# Running the NMDN model
print("Running NMDN model...")
tester = Tester("./data/exp_pl_534_run_2024-01-22_211045__480688")
tester.cfg.no_pkd_score = nmdn_only
tester.cfg.model.kano.kano_ckpt = None
model = tester.model
model.eval()
collate_fn = Collater(None, None)
data_processor = DataPostProcessor(tester.cfg)
loss_fn = MDNMixLossFn(tester.cfg)
loss_fn.compute_pkd_score = not nmdn_only
loss_fn.inference_mode()
out_info = defaultdict(lambda: [])
for i, lig_sdf in enumerate(tqdm(ligs, "NMDN model")):
    data = proc_hetero_graph(MPInfo(protein_pdb=prot, ligand_sdf=lig_sdf))
    data = data_processor(data, 0)
    data = collate_fn([data])
    data.prot_embed = prot_embed
    if not nmdn_only:
        data.mol_prop = phys_mol_prop[i]
        data.rmsd = torch.as_tensor([rmsd_info[lig_sdf]]).float().cuda()
    data.sample_id = torch.as_tensor([0]).cuda()
    pred = model(data)
    __, scores = loss_fn(pred, data, False, True)
    nmdn_score = scores["MDN_LOGSUM_DIST2_REFDIST2"].detach().cpu().item()
    
    out_info["lig"].append(lig_sdf)
    out_info["NMDN-Score"].append(nmdn_score)
    if not nmdn_only:
        pkd_score = scores["PROP_PRED"].detach().cpu().item()
        out_info["pKd-Score"].append(pkd_score)

out_df = pd.DataFrame(out_info)
print(out_df)
if save_csv is not None:
    out_df.to_csv(save_csv, index=False)
