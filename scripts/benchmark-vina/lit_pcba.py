from collections import defaultdict
from tempfile import TemporaryDirectory
import torch as th
import argparse
from tqdm import tqdm
from glob import glob
import os
import os.path as osp
import pandas as pd
import numpy as np


from geometry_processors.pl_dataset.lit_pcba_reader import LIT_PCBA_DiffDockReader, pdb_getter
from geometry_processors.process.vina_wrapper import VinaScoreCalculator
from utils.data.DummyIMDataset import DummyIMDataset

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

NMDN_TEST="/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688"
exp_id = osp.basename(NMDN_TEST).split("_run_")[0]

parser = argparse.ArgumentParser()
parser.add_argument("--target")
parser.add_argument("--array_id", type=int, default=None)
args = parser.parse_args()
target = args.target
array_id = args.array_id

pcba_reader = LIT_PCBA_DiffDockReader("/", target)
prot_pdb = pdb_getter[target]
prot = pcba_reader.pdb2polar_prot(prot_pdb)
if target == "ALDH1":
    prot = "/vast/sx801/geometries/LIT-PCBA-DiffDock/ALDH1/PolarH/5l2m.polar.pdb"
score_info = defaultdict(lambda: [])
nmdn_test_folder = glob(osp.join(NMDN_TEST, f"exp_pl_*_test_on_{target.lower()}-diffdock_*"))
assert len(nmdn_test_folder) == 1, nmdn_test_folder
nmdn_selected_fl = DummyIMDataset.nmdn_test2selected_fl(nmdn_test_folder[0])

fl2rank_mapper = {}
for selected_fl in nmdn_selected_fl:
    fl, rank = selected_fl.split(".")
    fl2rank_mapper[fl] = rank

lig_fls = glob(f"/{target}/pose_diffdock/raw_predicts/*")
if array_id is not None:
    lig_fls.sort()
    lig_fls = np.array_split(lig_fls, 20)[array_id]

tempdir = TemporaryDirectory()
calc = VinaScoreCalculator(tempdir)
for lig_fl in tqdm(lig_fls):
    fl = osp.basename(lig_fl)
    rank = fl2rank_mapper[fl] if fl in fl2rank_mapper else "1"
    lig = glob(osp.join(lig_fl, f"rank{rank}_confidence*.sdf"))
    if len(lig) == 0:
        continue
    lig = lig[0]
    try:
        res = calc.compute_score(prot, lig)
    except Exception as e:
        print(e)
        continue
    score_info["fl"].append(fl)
    for score_name in res.keys():
        score_info[f"{score_name}_affinity"].append(res[score_name])
        score_info[f"{score_name}_score"].append(res[score_name]/pKd2deltaG)
    # if len(score_info["fl"]) == 10: break

score_df = pd.DataFrame(score_info)
if array_id is None:
    score_df.to_csv(f"preds/{target}.csv", index=False)
else:
    os.makedirs(f"preds/{target}", exist_ok=True)
    score_df.to_csv(f"preds/{target}/{array_id}.csv", index=False)
