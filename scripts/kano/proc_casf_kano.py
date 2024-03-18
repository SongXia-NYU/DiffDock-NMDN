import argparse
import os
import os.path as osp
from typing import Dict

import rdkit
import pickle
from tqdm import tqdm
from chemprop_kano.features.featurization import MolGraph

from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader, mol2_to_info_list, polar_mol2_to_info
from geometry_processors.pl_dataset.conf_reader_factory import ConfReaderFactory

DROOT = "/CASF-2016-cyang"
reader = CASF2016Reader(DROOT)
DS_PROP = "lig.polar"
save_root = "/scratch/sx801/data/im_datasets/processed/"
mol_g_arg = argparse.Namespace(atom_messages=False)

def process_casf_scoring():
    ds_name = f"casf-scoring.{DS_PROP}.kano.pickle"
    ds: Dict[str, MolGraph] = {}
    info_list = []
    for pdb in reader.pdbs:
        prot_pdb = reader.pdb2prot_polarh(pdb)
        lig_mol2 = reader.pdb2lig_core_polarh(pdb)
        info_list.append(polar_mol2_to_info(prot_pdb, lig_mol2, pdb, pKd=reader.pdb2pkd(pdb)))
        if not osp.exists(prot_pdb) or not osp.exists(lig_mol2):
            print(prot_pdb)
            print(lig_mol2)
    for info in info_list:
        conf_reader, __ = ConfReaderFactory(info).get_lig_reader()
        mol = conf_reader.mol
        mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
        ds[info.labels["pdb"]] = mol_g
    with open(osp.join(save_root, ds_name), "wb") as f:
        pickle.dump(ds, f)



def process_casf_docking():
    info_list = []
    ds_name = f"casf-docking.{DS_PROP}.kano.pickle"

    for pdb in reader.pdbs:
        prot_pdb = reader.pdb2prot_polarh(pdb)
        mol2s = reader.pdb2dock_polarh_ligs(pdb)
        this_prot_list = []
        for mol2 in mol2s:
            this_prot_list.append(polar_mol2_to_info(prot_pdb, mol2, pdb))
        for info in this_prot_list[1:]:
            info.discard_p_info = True
        info_list.extend(this_prot_list)
        
    ds: Dict[str, MolGraph] = {}
    for info in tqdm(info_list):
        mol = info.mol
        mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
        ds[info.ligand_id] = mol_g
    with open(osp.join(save_root, ds_name), "wb") as f:
        pickle.dump(ds, f)


def process_casf_screening():
    ds_name = f"casf-screening.{DS_PROP}.kano"
    DST_ROOT = f"/scratch/sx801/data/im_datasets/processed/{ds_name}"
    os.makedirs(DST_ROOT, exist_ok=True)

    for i, pdb_tgt in enumerate(reader.screen_pdbs):
        save_pyg = osp.join(DST_ROOT, f"{pdb_tgt}.pyg")
        if osp.exists(save_pyg):
            print(f"Skipping: {pdb_tgt}")
            continue

        prot_pdb = reader.pdb2prot_polarh(pdb_tgt)
        info_list = []
        ds: Dict[str, MolGraph] = {}
        for pdb_lig in reader.pdbs:
            lig_mol2 = reader.pdb2screen_polarh_ligs(pdb_tgt, pdb_lig)
            info_list.extend(mol2_to_info_list(prot_pdb, lig_mol2, pdb_tgt))
        for info in info_list[1:]:
            info.discard_p_info = True
        for info in tqdm(info_list):
            mol = info.mol
            mol_g = MolGraph(None, mol, mol_g_arg, prompt=False)
            ds[info.ligand_id] = mol_g
        with open(osp.join(save_root, ds_name, f"{pdb_tgt}.pickle"), "wb") as f:
            pickle.dump(ds, f)



if __name__ == "__main__":
    # process_casf_docking()
    process_casf_scoring()
