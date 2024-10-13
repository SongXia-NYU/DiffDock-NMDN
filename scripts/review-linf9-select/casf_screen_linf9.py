from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd
from glob import glob
import os
import os.path as osp
import yaml
import argparse

from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.process.vina_wrapper import VinaScoreCalculator
from utils.scores.casf_blind_scores import score_rank_power

SCREENING_ROOT = "/screening/pose_diffdock/raw_predicts/"
reader = CASF2016BlindDocking("/")
screen_pdbs = reader.screen_pdbs
target_pdbs = list(set([pdb.split("_")[0] for pdb in screen_pdbs]))
target_pdbs.sort()

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
parser.add_argument("--score_name", type=str)
args = parser.parse_args()
array_id = args.array_id
SCORE_NAME = args.score_name
target_pdb = target_pdbs[array_id]

def compute_screening_score():
    # R in kcal/(mol.K)
    R = 1.98720425864083e-3
    logP_to_watOct = 2.302585093 * R * 298.15
    pKd2deltaG = -logP_to_watOct

    out_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir, scores=[SCORE_NAME])
    for pdb_root in tqdm(glob(f"{SCREENING_ROOT}/{target_pdb}_????")):
        for lig in glob(f"{pdb_root}/rank*_confidence*.sdf"):
            ligpdb = osp.basename(pdb_root).split("_")[-1]
            prot = f"/CASF-2016-cyang/coreset/{target_pdb}/{target_pdb}_protein.pdb"
            try:
                res = calc.compute_score(prot, lig)
            except Exception as e:
                print(e)
                continue
            if res is None: continue
            out_info["tgt_pdb"].append(target_pdb)
            out_info["lig_pdb"].append(ligpdb)
            out_info["lig_srcrank"].append(osp.basename(lig).split("_")[0].split("rank")[-1])
            out_info[f"{SCORE_NAME}_affinity"].append(res[SCORE_NAME])
            out_info[f"{SCORE_NAME}_score"].append(res[SCORE_NAME] / pKd2deltaG)

    out_df = pd.DataFrame(out_info)
    os.makedirs(f"./results_casf_screening/{SCORE_NAME}_raw_preds", exist_ok=True)
    out_df.to_csv(f"./results_casf_screening/{SCORE_NAME}_raw_preds/casf_screen_{SCORE_NAME}.{target_pdb}.csv", index=False)


if __name__ == "__main__":
    compute_screening_score()
