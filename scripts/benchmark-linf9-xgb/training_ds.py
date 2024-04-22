import argparse
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import json
from glob import glob
import os
import os.path as osp
import numpy as np

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/TrainingSet"
os.makedirs(protdir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int, default=None)
args = parser.parse_args()
array_id = args.array_id

def save_nmdn_res():
    nmdn_out_info = defaultdict(lambda: [])
    with open("training_ds.linf9.info.json") as f:
        ds_info = json.load(f)
    file_handle, protein_file, ligand_file = ds_info["file_handle"], ds_info["protein_file"], ds_info["ligand_file"]
    zipped_info = np.array_split(list(zip(file_handle, protein_file, ligand_file)), 100)[array_id]
    for fl, pf, lf in tqdm(zipped_info):
        if fl.startswith("nonbinders."):
            lf = lf.replace("/polarh/", "/raw_predicts/")
            lf_folder, rank = lf.split(".srcrank")
            rank = rank.split(".sdf")[0]
            lf = glob(osp.join(lf_folder, f"rank{rank}_confidence*.sdf"))
            assert len(lf) == 1, osp.join(lf_folder, f"rank{rank}_confidence*.sdf")
            lig = lf[0]

            prot = pf.replace("/polarh/", "/structures/")
        else:
            pdb, rank = osp.basename(lf).split("_")[0].split(".rank")
            lf_folder = f"/PDBBind2020_DiffDock-sampled/raw_predicts/{pdb}"
            lf = glob(osp.join(lf_folder, f"rank{rank}_confidence*.sdf"))
            assert len(lf) == 1, osp.join(lf_folder, f"rank{rank}_confidence*.sdf")
            lig = lf[0]
            prot = pf

        res = xgb_wrapper(prot, lig, protdir, linf9_only=True)
        if res is None: continue
        nmdn_out_info["file_handle"].append(fl)
        nmdn_out_info["linf9_score"].append(res["linf9_score"])
        # if len(nmdn_out_info["file_handle"]) > 10: break

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv(f"./preds/training_set/{array_id}.csv", index=False)

save_nmdn_res()
