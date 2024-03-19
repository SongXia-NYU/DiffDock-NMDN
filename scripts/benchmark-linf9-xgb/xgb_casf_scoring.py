from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from geometry_processors.linf9_xgb.linf9_xgb_wrapper import xgb_wrapper
from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"
prot = "/PDBBind2020_OG/refined-set/1a30/1a30_protein.pdb"

def save_nmdn_res():
    nmdn_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_nmdn_rank1(pdb)
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        nmdn_out_info["pdb"].append(pdb)
        nmdn_out_info["xgb_score"].append(res["xgb_score"])
        nmdn_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(nmdn_out_info)
    out_df.to_csv("./casf/nmdn_out.csv", index=False)

def save_diffdock_res():
    diffdock_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = reader.get_diffdock_rank1(pdb)
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        diffdock_out_info["pdb"].append(pdb)
        diffdock_out_info["xgb_score"].append(res["xgb_score"])
        diffdock_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(diffdock_out_info)
    out_df.to_csv("./casf/diffdock_out.csv", index=False)

def save_crystal_res():
    crystal_out_info = defaultdict(lambda: [])
    for pdb in tqdm(reader.dock_pdbs):
        prot = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_protein.pdb"
        lig = f"/CASF-2016-cyang/coreset/{pdb}/{pdb}_ligand.mol2"
        res = xgb_wrapper(prot, lig, protdir)
        if res is None: continue
        crystal_out_info["pdb"].append(pdb)
        crystal_out_info["xgb_score"].append(res["xgb_score"])
        crystal_out_info["linf9_score"].append(res["linf9_score"])

    out_df = pd.DataFrame(crystal_out_info)
    out_df.to_csv("./casf/crystal_out.csv", index=False)

save_diffdock_res()
save_nmdn_res()
save_crystal_res()
