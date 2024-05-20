from geometry_processors.process.sasa_calculator import SASA_PL_Calculator, SASASingleCalculator
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from utils.DataPrepareUtils import n_rotatable_bonds

info_df = pd.read_csv("/scratch/sx801/cache/rmsd_csv/JOB-pdbbind_yaowen_nonbinders.rmsd.csv")
nonbinder_df = info_df[info_df["file_handle"].map(lambda s: s.startswith("nonbinders."))]
pdbbind_df = info_df[info_df["file_handle"].map(lambda s: not s.startswith("nonbinders."))]


for i in tqdm(range(pdbbind_df.shape[0])):
    this_info = pdbbind_df.iloc[i]
    fl = this_info["file_handle"]
    pdb = fl.split(".")[0]
    fname = ".".join(fl.split(".")[1:]) + ".sdf"
    lig_file = f"/PDBBind2020_DiffDock-sampled/raw_predicts/{pdb}/{fname}"
    prot = f"/PDBBind2020_OG/RenumPDBs/{pdb}.renum.pdb"
    calc = SASA_PL_Calculator(prot, lig_file, f"/vast/sx801/temp_sasa/{fl}.pkl")
    calc.run()

nonbinder_info_df = pd.read_csv("/vast/sx801/geometries/Yaowen_nonbinders/sampled_pl_info.csv").set_index("file_handle")
for i in tqdm(range(nonbinder_df.shape[0])):
    this_info = nonbinder_df.iloc[i]
    lig_file = this_info["ligand_file"]
    lig_file = lig_file.replace("/vast/sx801/geometries/", "/")
    fl = this_info["file_handle"]
    fl_query = ".".join(fl.split(".")[:-1])
    pdb = nonbinder_info_df.loc[fl_query, "pdb_id"]
    prot = f"/vast/sx801/geometries/Yaowen_nonbinders/protein_crystal_pdbs/structures/{pdb}.pdb"
    calc = SASA_PL_Calculator(prot, lig_file, f"/vast/sx801/temp_sasa/{fl}.pkl")
    calc.run()
