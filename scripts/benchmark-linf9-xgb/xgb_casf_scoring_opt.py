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

reader = CASF2016BlindDocking("dummy")
protdir = "/scratch/sx801/temp/LinF9_xgb_temp/CASF-2016"

if osp.exists("/scratch/sx801/"):
    SCRIPT_ROOT = "/softwares"
    DS_ROOT = "/vast/sx801"
else:
    SCRIPT_ROOT = "/home/carrot_of_rivia/Documents/PycharmProjects"
    DS_ROOT = "/home/carrot_of_rivia/Documents/disk/datasets"
PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor"
PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_ligand"
LINF9_PATH = "/softwares/Lin_F9_test"
LINF9 = f"{LINF9_PATH}/smina.static"

def handler(signum, frame):
    raise RuntimeError("TIMEOUT running PREPARE_LIG")

class LinF9LocalOptimizer:
    def __init__(self, protein_pdb=None, ligand_sdf=None, ligand_linf9_opt=None, 
                 ligand_mol2=None, protein_pdbqt=None, ligand_pdbqt=None) -> None:
        self.protein_pdb = protein_pdb
        self.ligand_mol2 = ligand_mol2
        self.ligand_sdf = ligand_sdf

        self.protein_pdbqt = protein_pdbqt
        self.ligand_pdbqt = ligand_pdbqt
        self.ligand_linf9_opt = ligand_linf9_opt

    def run(self):
        temp_dir = TemporaryDirectory()
        if self.protein_pdbqt is None:
            self.protein_pdbqt = osp.join(temp_dir.name, "protein.pdbqt")
        if self.ligand_pdbqt is None:
            self.ligand_pdbqt = osp.join(temp_dir.name, "ligand.pdbqt")
        if not osp.exists(self.protein_pdbqt):
            assert self.protein_pdb is not None
            subprocess.run(f"{PREPARE_PROT} -r {self.protein_pdb} -U nphs_lps -A 'checkhydrogens' -o {self.protein_pdbqt} ", shell=True, check=True)
        if not osp.exists(self.ligand_pdbqt):
            ligand_mol2 = self.ligand_mol2
            if self.ligand_mol2 is None:
                assert self.ligand_sdf is not None
                mol2_file = osp.join(temp_dir.name, "tmp.mol2")
                conv_cmd = f"obabel -isdf {self.ligand_sdf} -omol2 -O {mol2_file}"
                print(conv_cmd)
                subprocess.run(conv_cmd, shell=True, check=True)
                ligand_mol2 = mol2_file
            ligand_dir = osp.dirname(ligand_mol2)
            lig_cmd = f"{PREPARE_LIG} -l {ligand_mol2} -U nphs_lps -A 'checkhydrogens' -o {self.ligand_pdbqt} "
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)
            subprocess.run(lig_cmd, shell=True, check=True, cwd=ligand_dir)
            signal.alarm(0)
        linf9_cmd = f"{LINF9} -r {self.protein_pdbqt} -l {self.ligand_pdbqt} --local_only --scoring Lin_F9 -o {self.ligand_linf9_opt} "
        subprocess.run(linf9_cmd, shell=True, check=True)
        temp_dir.cleanup()

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
