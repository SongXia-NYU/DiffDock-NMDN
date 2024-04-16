from collections import defaultdict
import signal
import subprocess
from tempfile import TemporaryDirectory
from tqdm import tqdm
import pandas as pd
import os.path as osp
from glob import glob

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.pl_dataset.linf9_local_optimizer import LinF9LocalOptimizer

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

def save_nmdn_res():
    nmdn_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_docking_nmdn_rank1(pdb)
        lig_opt = osp.join("/scratch/sx801/scripts/DiffDock-NMDN/scripts/benchmark-linf9-xgb/casf_opt/temp",
                           f"{pdb}.{osp.basename(lig)}".replace(".sdf", ".pdb"))
        if not osp.exists(lig_opt):
            try:
                opt = LinF9LocalOptimizer(ligand_sdf=lig, protein_pdb=prot, ligand_linf9_opt=lig_opt)
                opt.run()
            except Exception as e:
                print(f"Error proc {pdb}: {e}")
                continue
        res = xgb_wrapper(prot, lig_opt, protdir, False)
        if res is None: continue
        nmdn_out_info["pdb"].append(pdb)
        nmdn_out_info["xgb_score"].append(res["xgb_score"])
        nmdn_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./casf_opt/nmdn_out.csv", index=False)

def check_errors():
    success_df = pd.read_csv("./casf_opt/nmdn_out.csv")
    success_pdbs = set(success_df["pdb"].values.reshape(-1).tolist())
    allpdbs = glob(osp.join("/scratch/sx801/scripts/DiffDock-NMDN/scripts/benchmark-linf9-xgb/casf_opt/temp", "*.pdb"))
    for pdb in allpdbs:
        pdb_id = osp.basename(pdb).split(".")[0]
        if pdb_id in success_pdbs:
            continue
        prot = f"/CASF-2016-cyang/coreset/{pdb_id}/{pdb_id}_protein.pdb"
        breakpoint()
        res = xgb_wrapper(prot, pdb, protdir, False)
        print(res)


# save_diffdock_res()
save_nmdn_res()
# save_crystal_res()
# check_errors()
