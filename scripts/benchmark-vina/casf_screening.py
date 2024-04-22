import argparse
from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd
import numpy as np

from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.process.vina_wrapper import VinaScoreCalculator
# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id: int = args.array_id

reader = CASF2016BlindDocking("/")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

res_info = defaultdict(lambda: [])

screen_pdbs = reader.screen_pdbs
screen_pdbs.sort()
screen_pdbs = np.array_split(screen_pdbs, 10)[array_id]

temp_dir = TemporaryDirectory()
calc = VinaScoreCalculator(temp_dir)
for idx, screen_pdb in enumerate(tqdm(screen_pdbs)):
    tgt_pdb, lig_pdb = screen_pdb.split("_")
    prot = f"/CASF-2016-cyang/coreset/{tgt_pdb}/{tgt_pdb}_protein.pdb"

    # res_diffdock = xgb_wrapper(prot, diffdock_rank1, protdir)
    # if res_diffdock is None: res_diffdock = {"xgb_score": "", "linf9_score": ""}
    try:
        nmdn_rank1 = reader.get_screening_nmdn_rank1(tgt_pdb, lig_pdb)
        res = calc.compute_score(prot, nmdn_rank1)
    except Exception as e:
        print(e)
        continue

    res_info["tgt_pdb"].append(tgt_pdb)
    res_info["lig_pdb"].append(lig_pdb)
    for score_name in res.keys():
        res_info[f"{score_name}_affinity"].append(res[score_name])
        res_info[f"{score_name}_score"].append(res[score_name]/pKd2deltaG)
    # if idx==10: break
    # res_info["diffdock_xgb_score"].append(res_diffdock["xgb_score"])
    # res_info["diffdock_linf9_score"].append(res_diffdock["linf9_score"])

res_df = pd.DataFrame(res_info)
res_df.to_csv(f"./preds/casf-screening/array.{array_id}.csv", index=False)
