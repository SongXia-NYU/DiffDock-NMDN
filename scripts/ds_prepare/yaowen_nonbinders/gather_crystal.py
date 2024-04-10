"""get crystal target structures"""

import pandas as pd
from tqdm import tqdm
from Bio.PDB import parse_pdb_header
from urllib import request
from urllib.error import HTTPError
from glob import glob
import os.path as osp
from geometry_processors.pl_dataset.yaowen_nonbinder_reader import NonbinderReader

# 909 IDs were mapped to 30,878 results
# 290 ID were not mapped
def gen_uniprot_ids():
    reader = NonbinderReader()
    print(",".join(reader.uniprot_ids))

# crystal pose resolution
# download pdb headers only
def gather_crystal_res():
    pdb_df = pd.read_csv("/vast/sx801/geometries/Yaowen_nonbinders/idmapping_2024_04_07.tsv", sep="\t")
    pdbs = set(pdb_df["To"].values.reshape(-1).tolist())
    for pdb in tqdm(pdbs):
        save_f = f"/vast/sx801/geometries/Yaowen_nonbinders/protein_crystal_pdbs/headers/{pdb}.header.pdb"
        if osp.exists(save_f): continue
        try:
            request.urlretrieve(f"https://files.rcsb.org/header/{pdb}.pdb",
                                save_f)
        except HTTPError as e:
            print(e)

def parse_res_info():
    info_list = []
    interested_keys = ["deposition_date", "resolution", "release_date", "structure_method"]
    for header_pdb in glob("/vast/sx801/geometries/Yaowen_nonbinders/protein_crystal_pdbs/headers/*.header.pdb"):
        pdb_id = osp.basename(header_pdb).split(".")[0]
        header_info = parse_pdb_header(header_pdb)
        header_info = {key: header_info[key] if key in header_info else "" for key in interested_keys}
        hedaer_df = pd.DataFrame(header_info, index=[pdb_id])
        info_list.append(hedaer_df)
    info_df = pd.concat(info_list)
    info_df.to_csv(f"/vast/sx801/geometries/Yaowen_nonbinders/protein_crystal_pdbs/res_info.csv")

def get_crystal_structure():
    reader = NonbinderReader()
    pdbs = set(reader.sampled_pl_info_df["pdb_id"].values.reshape(-1).tolist())
    for pdb in tqdm(pdbs):
        save_f = f"/vast/sx801/geometries/Yaowen_nonbinders/protein_crystal_pdbs/structures/{pdb}.pdb"
        try:
            request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb",
                                save_f)
        except HTTPError as e:
            print(e)

if __name__ == "__main__":
    get_crystal_structure()
