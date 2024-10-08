from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.process.vina_wrapper import VinaScoreCalculator

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

def save_nmdn_res():
    nmdn_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_docking_nmdn_rank1(pdb)
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        nmdn_out_info["pdb"].append(pdb)
        nmdn_out_info["xgb_score"].append(res["xgb_score"])
        nmdn_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./casf/nmdn_out.csv", index=False)

def save_nmdn_planb_res():
    # R in kcal/(mol.K)
    R = 1.98720425864083e-3
    logP_to_watOct = 2.302585093 * R * 298.15
    pKd2deltaG = -logP_to_watOct

    nmdn_out_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir, scores=["Lin_F9"])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        try:
            lig = reader.get_docking_nmdn_rank1(pdb)
            res = calc.compute_score(prot, lig)
        except Exception as e:
            print(e)
            continue
        if res is None: continue
        nmdn_out_info["pdb"].append(pdb)
        nmdn_out_info["linf9_affinity"].append(res["Lin_F9"])
        nmdn_out_info["linf9_score"].append(res["Lin_F9"] / pKd2deltaG)

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./casf/nmdn_planb_out.csv", index=False)

def save_diffdock_res():
    diffdock_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_docking_diffdock_rank1(pdb)
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        diffdock_out_info["pdb"].append(pdb)
        diffdock_out_info["xgb_score"].append(res["xgb_score"])
        diffdock_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(diffdock_out_info)
    out_df.to_csv("./casf/diffdock_out.csv", index=False)

def save_crystal_res():
    crystal_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_ligand.mol2"
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        crystal_out_info["pdb"].append(pdb)
        crystal_out_info["xgb_score"].append(res["xgb_score"])
        crystal_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(crystal_out_info)
    out_df.to_csv("./casf/crystal_out.csv", index=False)

save_nmdn_planb_res()
