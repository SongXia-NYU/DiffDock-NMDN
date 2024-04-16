import argparse
from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd
import numpy as np
import os.path as osp

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.pl_dataset.linf9_local_optimizer import LinF9LocalOptimizer

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
for idx, screen_pdb in enumerate(tqdm(screen_pdbs)):
    tgt_pdb, lig_pdb = screen_pdb.split("_")
    prot = f"/CASF-2016-cyang/coreset/{tgt_pdb}/{tgt_pdb}_protein.pdb"
    try:
        # diffdock_rank1 = reader.get_screening_diffdock_rank1(tgt_pdb, lig_pdb)
        nmdn_rank1 = reader.get_screening_nmdn_rank1(tgt_pdb, lig_pdb)
        lig_opt = osp.join(temp_dir.name, f"{tgt_pdb}.{lig_pdb}.{osp.basename(nmdn_rank1)}".replace(".sdf", ".pdb"))
        opt = LinF9LocalOptimizer(ligand_sdf=nmdn_rank1, protein_pdb=prot, ligand_linf9_opt=lig_opt)
        opt.run()
    except Exception as e:
        print("Error proc ", screen_pdb, ": ", str(e))
        continue

    # res_diffdock = xgb_wrapper(prot, diffdock_rank1, protdir)
    # if res_diffdock is None: res_diffdock = {"xgb_score": "", "linf9_score": ""}
    res_nmdn = xgb_wrapper(prot, lig_opt, protdir)
    if res_nmdn is None: res_nmdn = {"xgb_score": "", "linf9_score": ""}

    res_info["tgt_pdb"].append(tgt_pdb)
    res_info["lig_pdb"].append(lig_pdb)
    res_info["nmdn_xgb_score"].append(res_nmdn["xgb_score"])
    res_info["nmdn_linf9_score"].append(res_nmdn["linf9_score"])
    # if idx==10: break
    # res_info["diffdock_xgb_score"].append(res_diffdock["xgb_score"])
    # res_info["diffdock_linf9_score"].append(res_diffdock["linf9_score"])

res_df = pd.DataFrame(res_info)
res_df.to_csv(f"casf_opt/screening/arrays/array.{array_id}.csv", index=False)
