from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd

from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.process.vina_wrapper import VinaScoreCalculator

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

def save_nmdn_res():
    nmdn_out_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir)
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_docking_nmdn_rank1(pdb)
        try:
            res = calc.compute_score(prot, lig)
        except Exception as e:
            print(e)
            continue
        nmdn_out_info["pdb"].append(pdb)
        for score_name in res.keys():
            nmdn_out_info[f"{score_name}_affinity"].append(res[score_name])
            nmdn_out_info[f"{score_name}_score"].append(res[score_name]/pKd2deltaG)

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./preds/casf-scoring.csv", index=False)

def save_crystal():
    nmdn_out_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir)
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_ligand.mol2"
        try:
            res = calc.compute_score(prot, lig)
        except Exception as e:
            print(e)
            continue
        nmdn_out_info["pdb"].append(pdb)
        nmdn_out_info["vina_affinity"].append(res)
        nmdn_out_info["vina_score"].append(res/pKd2deltaG)

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./preds/casf-scoring-crystal.csv", index=False)

save_nmdn_res()
