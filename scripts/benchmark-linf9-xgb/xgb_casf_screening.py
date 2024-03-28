import argparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id: int = args.array_id

reader = CASF2016BlindDocking("/")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

res_info = defaultdict(lambda: [])

screen_pdbs = reader.screen_pdbs
screen_pdbs.sort()
screen_pdbs = np.array_split(screen_pdbs, 100)[array_id]

for screen_pdb in tqdm(screen_pdbs):
    tgt_pdb, lig_pdb = screen_pdb.split("_")
    prot = f"/CASF-2016-cyang/coreset/{tgt_pdb}/{tgt_pdb}_protein.pdb"
    try:
        diffdock_rank1 = reader.get_screening_diffdock_rank1(tgt_pdb, lig_pdb)
        nmdn_rank1 = reader.get_screening_nmdn_rank1(tgt_pdb, lig_pdb)
    except Exception as e:
        print("Error proc ", screen_pdb, ": ", str(e))
        continue

    res_diffdock = xgb_wrapper(prot, diffdock_rank1, protdir)
    if res_diffdock is None: res_diffdock = {"xgb_score": "", "linf9_score": ""}
    res_nmdn = xgb_wrapper(prot, nmdn_rank1, protdir)
    if res_nmdn is None: res_nmdn = {"xgb_score": "", "linf9_score": ""}

    res_info["tgt_pdb"].append(tgt_pdb)
    res_info["lig_pdb"].append(lig_pdb)
    res_info["nmdn_xgb_score"].append(res_nmdn["xgb_score"])
    res_info["nmdn_linf9_score"].append(res_nmdn["linf9_score"])
    res_info["diffdock_xgb_score"].append(res_diffdock["xgb_score"])
    res_info["diffdock_linf9_score"].append(res_diffdock["linf9_score"])

res_df = pd.DataFrame(res_info)
res_df.to_csv(f"casf/screening/array.{array_id}.csv", index=False)
