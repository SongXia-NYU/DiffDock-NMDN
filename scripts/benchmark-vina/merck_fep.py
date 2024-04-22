from tempfile import TemporaryDirectory
from collections import defaultdict
import os.path as osp
import pandas as pd
from tqdm import tqdm
from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader
from geometry_processors.process.vina_wrapper import VinaScoreCalculator

# R in kcal/(mol.K)
R = 1.98720425864083e-3
logP_to_watOct = 2.302585093 * R * 298.15
pKd2deltaG = -logP_to_watOct

protdir = "/scratch/sx801/temp/LinF9_xgb_temp/MerckFEP"
reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")

def opt():
    nmdn_info = defaultdict(lambda: [])
    tempdir = TemporaryDirectory()
    calc = VinaScoreCalculator(tempdir)
    for diffdock_root in tqdm(reader.get_ligs_diffdock_root()):
        file_handle = osp.basename(diffdock_root)
        target = file_handle.split(".")[0]
        lig_id = ".".join(file_handle.split(".")[1:])
        lig_nmdn_rank1 = reader.get_nmdn_rank1(target, lig_id)
        prot = reader.get_prot_src(target)

        try:
            res = calc.compute_score(prot, lig_nmdn_rank1)
        except Exception as e:
            print(e)
            continue
        nmdn_info["file_handle"].append(f"{target}.{lig_id}")
        for score_name in res.keys():
            nmdn_info[f"{score_name}_affinity"].append(res[score_name])
            nmdn_info[f"{score_name}_score"].append(res[score_name]/pKd2deltaG)

    out_df = pd.DataFrame(nmdn_info)
    out_df.to_csv("./preds/merck-fep.csv", index=False)
    tempdir.cleanup()

opt()
