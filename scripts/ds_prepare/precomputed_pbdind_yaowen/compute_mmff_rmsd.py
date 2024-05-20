from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from utils.rmsd import compute_mmff_rmsd

info_df = pd.read_csv("/scratch/sx801/cache/rmsd_csv/JOB-pdbbind_yaowen_nonbinders.rmsd.csv")
nonbinder_df = info_df[info_df["file_handle"].map(lambda s: s.startswith("nonbinders."))]
pdbbind_df = info_df[info_df["file_handle"].map(lambda s: not s.startswith("nonbinders."))]

precomputed_pbind_df = pd.read_csv("/scratch/sx801/scripts/Mol3DGenerator/scripts/mmff_optimize/pdbbind2020_og.rmsd.csv")
precomputed_pbind_df = precomputed_pbind_df.set_index("pdb")

rmsd_info = defaultdict(lambda: [])
for i in tqdm(range(pdbbind_df.shape[0])):
    this_info = pdbbind_df.iloc[i]
    fl = this_info["file_handle"]
    pdb = fl.split(".")[0]
    try:
        rmsd = precomputed_pbind_df.loc[pdb, "rmsd"]
    except KeyError:
        rmsd = None
    rmsd_info["file_handle"].append(fl)
    rmsd_info["rmsd"].append(rmsd)

for i in tqdm(range(nonbinder_df.shape[0])):
    this_info = nonbinder_df.iloc[i]
    lig_file = this_info["ligand_file"]
    lig_file = lig_file.replace("/vast/sx801/geometries/", "/")
    fl = this_info["file_handle"]
    rmsd = compute_mmff_rmsd(lig_file)
    rmsd_info["file_handle"].append(fl)
    rmsd_info["rmsd"].append(rmsd)

rmsd_df = pd.DataFrame(rmsd_info)
rmsd_df.to_csv("/scratch/sx801/cache/rmsd_csv/pdbbind_yaowen_nonbinders.rmsd.csv", index=False)
