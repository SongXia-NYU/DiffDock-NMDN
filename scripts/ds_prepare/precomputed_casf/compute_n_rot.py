from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from glob import glob
import os.path as osp

from utils.DataPrepareUtils import n_rotatable_bonds

for job_csv in glob("/scratch/sx801/cache/nrot_csv/casf2016-blind-screening/JOB-*.csv"):
    job_df = pd.read_csv(job_csv)

    rmsd_info = defaultdict(lambda: [])

    for i in tqdm(range(job_df.shape[0]), osp.basename(job_csv)):
        this_info = job_df.iloc[i]
        lig_file = this_info["ligand_file"]
        fl = this_info["file_handle"]
        lig_rank = fl.split("rank")[-1]
        lig_file = osp.join(osp.dirname(lig_file).replace("polarh", "raw_predicts"), f"rank{lig_rank}_confidence*.sdf")
        lig_file = glob(lig_file)[0]
        rmsd_info["file_handle"].append(fl)
        rmsd_info["n_rotatable_bond"].append(n_rotatable_bonds(lig_file))

    rmsd_df = pd.DataFrame(rmsd_info)
    rmsd_df.to_csv(job_csv.replace("JOB-", ""), index=False)
