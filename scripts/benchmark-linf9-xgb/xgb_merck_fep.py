from collections import defaultdict
import os.path as osp
import pandas as pd
from tqdm import tqdm
from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader

protdir = "/scratch/sx801/temp/LinF9_xgb_temp/MerckFEP"
reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")

nmdn_info = defaultdict(lambda: [])
diffdock_info = defaultdict(lambda: [])
for diffdock_root in tqdm(reader.get_ligs_diffdock_root()):
    file_handle = osp.basename(diffdock_root)
    target = file_handle.split(".")[0]
    lig_id = ".".join(file_handle.split(".")[1:])
    lig_nmdn_rank1 = reader.get_nmdn_rank1(target, lig_id)
    prot = reader.get_prot_src(target)

    res = xgb_wrapper(prot, lig_nmdn_rank1, protdir)
    if res is None: continue
    nmdn_info["file_handle"].append(f"{target}.{lig_id}")
    nmdn_info["xgb_score"].append(res["xgb_score"])
    nmdn_info["linf9_score"].append(res["linf9_score"])

out_df = pd.DataFrame(nmdn_info)
out_df.to_csv("./merck-fep/nmdn_out.csv", index=False)

for diffdock_root in tqdm(reader.get_ligs_diffdock_root()):
    file_handle = osp.basename(diffdock_root)
    target = file_handle.split(".")[0]
    lig_id = ".".join(file_handle.split(".")[1:])
    lig_diffdock_rank1 = reader.get_diffdock_rank1(target, lig_id)
    prot = reader.get_prot_src(target)

    res = xgb_wrapper(prot, lig_diffdock_rank1, protdir)
    if res is None: continue
    diffdock_info["file_handle"].append(f"{target}.{lig_id}")
    diffdock_info["xgb_score"].append(res["xgb_score"])
    diffdock_info["linf9_score"].append(res["linf9_score"])

out_df = pd.DataFrame(diffdock_info)
out_df.to_csv("./merck-fep/diffdock_out.csv", index=False)
