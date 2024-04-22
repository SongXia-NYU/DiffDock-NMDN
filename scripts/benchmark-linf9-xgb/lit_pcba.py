from collections import defaultdict
import json
import argparse
import time
from tqdm import tqdm
from glob import glob
import os
import os.path as osp
import pandas as pd
import numpy as np


from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

NMDN_TEST="/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688"
exp_id = osp.basename(NMDN_TEST).split("_run_")[0]

parser = argparse.ArgumentParser()
parser.add_argument("--target")
parser.add_argument("--array_id", type=int, default=None)
parser.add_argument("--info_gen", action="store_true")
args = parser.parse_args()
target = args.target
array_id = args.array_id

def info_gen():
    from geometry_processors.pl_dataset.lit_pcba_reader import LIT_PCBA_DiffDockReader, pdb_getter
    from utils.data.DummyIMDataset import DummyIMDataset

    if array_id is not None and array_id != 0:
        # waiting for other programs to finish the info generation
        total_sleeped = 0
        while total_sleeped < 300:
            time.sleep(5)
            total_sleeped += 5
            if osp.exists(f"preds/INFO.{target}.json"):
                break
        return
    
    pcba_reader = LIT_PCBA_DiffDockReader("/", target)
    prot_pdb = pdb_getter[target]
    prot = pcba_reader.pdb2polar_prot(prot_pdb)
    if target == "ALDH1":
        prot = "/vast/sx801/geometries/LIT-PCBA-DiffDock/ALDH1/PolarH/5l2m.polar.pdb"
    
    nmdn_test_folder = glob(osp.join(NMDN_TEST, f"exp_pl_*_test_on_{target.lower()}-diffdock_*"))
    assert len(nmdn_test_folder) == 1, nmdn_test_folder
    nmdn_selected_fl = DummyIMDataset.nmdn_test2selected_fl(nmdn_test_folder[0])

    fl2rank_mapper = {}
    for selected_fl in nmdn_selected_fl:
        fl, rank = selected_fl.split(".")
        fl2rank_mapper[fl] = rank

    lig_fls = glob(f"/{target}/pose_diffdock/raw_predicts/*")

    info_list = []
    for lig_fl in tqdm(lig_fls):
        fl = osp.basename(lig_fl)
        rank = fl2rank_mapper[fl] if fl in fl2rank_mapper else "1"
        lig = glob(osp.join(lig_fl, f"rank{rank}_confidence*.sdf"))
        info_list.append((fl, lig))
    with open(f"preds/INFO.{target}.json", "w") as f:
        json.dump({"prot": prot, "info_list": info_list}, f)

def linf9_xgb():
    score_info = defaultdict(lambda: [])
    protdir = "/scratch/sx801/temp/LinF9_xgb_temp/LIT-PCBA"
    os.makedirs(protdir, exist_ok=True)
    with open(f"preds/INFO.{target}.json", "r") as f:
        info = json.load(f)
    prot = info["prot"]
    info_list = info["info_list"]
    if array_id is not None:
        info_list = np.array_split(info_list, 10)[array_id]
    for fl, lig in tqdm(info_list):
        if len(lig) == 0:
            continue
        lig = lig[0]
        try:
            res = xgb_wrapper(prot, lig, protdir, False)
        except Exception as e:
            print(e)
            continue
        if res is None: continue
        score_info["fl"].append(fl)
        score_info["xgb_score"].append(res["xgb_score"])
        score_info["linf9_score"].append(res["linf9_score"])
        # if len(score_info["fl"]) == 10: break

    score_df = pd.DataFrame(score_info)
    if array_id is None:
        score_df.to_csv(f"preds/{target}.csv", index=False)
    else:
        os.makedirs(f"preds/{target}", exist_ok=True)
        score_df.to_csv(f"preds/{target}/{array_id}.csv", index=False)

if args.info_gen: info_gen()
else: linf9_xgb()
