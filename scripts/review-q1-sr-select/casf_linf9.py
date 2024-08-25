from collections import defaultdict
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd
from glob import glob
import os.path as osp
import yaml
import argparse

from geometry_processors.process.vina_wrapper import VinaScoreCalculator
from utils.scores.casf_blind_scores import score_rank_power

DOCKING_ROOT = "/scratch/sx801/scripts/raw_data/diffdock/test/infer_casf_dock021_dock022_2023-06-25_002147/raw_predicts"

parser = argparse.ArgumentParser()
parser.add_argument("--score_name", type=str)
args = parser.parse_args()
SCORE_NAME = args.score_name

def compute_docking_score():
    # R in kcal/(mol.K)
    R = 1.98720425864083e-3
    logP_to_watOct = 2.302585093 * R * 298.15
    pKd2deltaG = -logP_to_watOct

    out_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir, scores=[SCORE_NAME])
    for pdb_root in tqdm(glob(f"{DOCKING_ROOT}/????")):
        for lig in glob(f"{pdb_root}/rank*_confidence*.sdf"):
            pdb = osp.basename(pdb_root)
            prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
            try:
                res = calc.compute_score(prot, lig)
            except Exception as e:
                print(e)
                continue
            if res is None: continue
            out_info["pdb"].append(pdb)
            out_info["lig_srcrank"].append(osp.basename(lig).split("_")[0].split("rank")[-1])
            out_info[f"{SCORE_NAME}_affinity"].append(res[SCORE_NAME])
            out_info[f"{SCORE_NAME}_score"].append(res[SCORE_NAME] / pKd2deltaG)

    out_df = pd.DataFrame(out_info)
    out_df.to_csv(f"./results_casf_docking/casf_{SCORE_NAME}.csv", index=False)


if __name__ == "__main__":
    compute_docking_score()
