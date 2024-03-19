import json
import os
import os.path as osp
from typing import Dict, List, Set
import pandas as pd

from glob import glob

import torch
from tqdm import tqdm
import yaml
from geometry_processors.lazy_property import lazy_property

from geometry_processors.pl_dataset.prot_utils import pdb2chain_seqs, pdb2seq

class PDB2020DSReader:
    def __init__(self, ds_root: str, phys_infuse_root: str = None) -> None:
        self.ds_root = ds_root
        self.phys_infuse_root = phys_infuse_root
        self.refined_set_root = osp.join(ds_root, "refined-set")
        self.others_root = osp.join(ds_root, "v2020-other-PL")
        self.info_csv = osp.join(self.refined_set_root, "index", "INDEX_general_PL_data.2020")
        self.linf9_csv = osp.join(ds_root, "OG_LinF9_RMSD.csv")
        self.renum_pdb_root = osp.join(self.ds_root, "RenumPDBs")
        self.polar_pdb_root = osp.join(self.ds_root, "PolarPDBs")
        self.pdbqt_root = osp.join(self.ds_root, "PDBQT")
        self.protein_pdbqt_root = osp.join(self.pdbqt_root, "protein")
        self.ligand_pdbqt_root = osp.join(self.pdbqt_root, "ligand")
        self.linf9_opt_root = osp.join(self.ds_root, "Lin_F9_OPT")
        self.polar_root = osp.join(self.ds_root, "PolarH")
        self.info_root = osp.join(self.ds_root, "info")
        self.noh_root = osp.join(self.ds_root, "noH")
        self.prot_noh_root = osp.join(self.noh_root, "protein")
        self.lig_proton_noh_root = osp.join(self.noh_root, "lig_proton")
        self.martini_root = osp.join(self.ds_root, "Martini")

        self._refined_pdbs = None
        self._others_pdbs = None
        self._info_df = None
        self._linf9_df = None
        self._pdb2pkd = None
        self._ds_size = None
        self._polar_seq_info: Dict[str, str] = None

    @property
    def polar_seq_info(self):
        if self._polar_seq_info is not None:
            return self._polar_seq_info
        
        polar_seq_yaml = osp.join(self.info_root, "polar_seq.yaml")
        if osp.exists(polar_seq_yaml):
            with open(polar_seq_yaml) as f:
                self._polar_seq_info = yaml.safe_load(f)
            return self._polar_seq_info
        
        seq_info = {}
        for pdb in tqdm(self.iter_pdbs(), total=len(self)):
            prot = self.pdb2polar_prot(pdb)
            if not osp.exists(prot):
                seq_info[pdb] = None
                continue
            seq = pdb2seq(prot)
            seq_info[pdb] = seq
        with open(polar_seq_yaml, "w") as f:
            yaml.safe_dump(seq_info, f)
        self._polar_seq_info = seq_info
        return self._polar_seq_info
    
    @lazy_property
    def polar_chains_seqs_info(self):
        seq_info = {}
        for pdb in tqdm(self.iter_pdbs(), total=len(self)):
            prot = self.pdb2polar_prot(pdb)
            if not osp.exists(prot):
                seq_info[pdb] = None
                continue
            seq = pdb2chain_seqs(prot)
            seq_info[pdb] = seq
        return seq_info

    def iter_refined_set(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.refined_set_root, pdb), pdb
        
    def iter_others_set(self):
        for pdb in self.others_pdbs:
            yield osp.join(self.others_root, pdb), pdb

    def iter_pdb_roots(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.refined_set_root, pdb), pdb
        for pdb in self.others_pdbs:
            yield osp.join(self.others_root, pdb), pdb

    def pdb2renum_file(self, pdb):
        return osp.join(self.renum_pdb_root, f"{pdb}.renum.pdb")
    
    def pdb2polar_prot(self, pdb: str):
        return osp.join(self.polar_pdb_root, f"{pdb}.polar.pdb")

    def pdb2martini_file(self, pdb):
        return osp.join(self.martini_root, "pdb", f"{pdb}.renum.martini.pdb")
    
    def pdb2noh_prot(self, pdb: str):
        return osp.join(self.prot_noh_root, f"{pdb}.noh.pdb")
    
    def pdb2noh_lig_proton(self, pdb: str):
        return osp.join(self.lig_proton_noh_root, f"{pdb}_ligand.noh.mol2")

    def iter_martini_pdbs(self):
        for pdb in self.iter_pdbs():
            yield self.pdb2martini_file(pdb)

    def iter_renum_pdbs(self):
        for pdb in self.refined_pdbs:
            yield self.pdb2renum_file(pdb)
        for pdb in self.others_pdbs:
            yield self.pdb2renum_file(pdb)
    
    def iter_pdbs(self):
        for pdb in self.refined_pdbs:
            yield pdb
        for pdb in self.others_pdbs:
            yield pdb

    @property
    def iter_src_pdbs(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.refined_set_root, pdb, f"{pdb}_protein.pdb")
        for pdb in self.others_pdbs:
            yield osp.join(self.others_root, pdb, f"{pdb}_protein.pdb")

    @property
    def iter_proton_ligs(self):
        """
        *Importantly*, the ligand molecule in the Mol2 format is presented in
        the protonation form as described above; but the ligand molecule in the
        SDF format is presented in its neutral form (e.g.-COOH). This is a new
        feature starting from PDBbind version 2020, which aims at providing more
        options. The users should decide which form to use for their own study.

        --From the readme file 'readme/PDBbind-101.txt' in PDBBind2020
        """
        for pdb in self.refined_pdbs:
            yield osp.join(self.refined_set_root, pdb, f"{pdb}_ligand.mol2")
        for pdb in self.others_pdbs:
            yield osp.join(self.others_root, pdb, f"{pdb}_ligand.mol2")

    def pdb2proton_lig(self, pdb: str):
        if pdb in self.refined_pdbs: 
            return osp.join(self.refined_set_root, pdb, f"{pdb}_ligand.mol2")
        return osp.join(self.others_root, pdb, f"{pdb}_ligand.mol2")
    
    def pdb2neutral_lig(self, pdb: str):
        if pdb in self.refined_pdbs: 
            return osp.join(self.refined_set_root, pdb, f"{pdb}_ligand.sdf")
        return osp.join(self.others_root, pdb, f"{pdb}_ligand.sdf")

    @property
    def iter_neutral_ligs(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.refined_set_root, pdb, f"{pdb}_ligand.sdf")
        for pdb in self.others_pdbs:
            yield osp.join(self.others_root, pdb, f"{pdb}_ligand.sdf")

    @property
    def iter_proton_polar_ligs(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.polar_root, "ligands", "protonated", f"{pdb}_ligand.mol2")
        for pdb in self.others_pdbs:
            yield osp.join(self.polar_root, "ligands", "protonated", f"{pdb}_ligand.mol2")

    def pdb2proton_polar_lig(self, pdb: str):
        return osp.join(self.polar_root, "ligands", "protonated", f"{pdb}_ligand.mol2")

    @property
    def iter_neutral_polar_ligs(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.polar_root, "ligands", "neutral", f"{pdb}_ligand.sdf")
        for pdb in self.others_pdbs:
            yield osp.join(self.polar_root, "ligands", "neutral", f"{pdb}_ligand.sdf")

    def pdb2neutral_polar_lig(self, pdb: str):
        return osp.join(self.polar_root, "ligands", "neutral", f"{pdb}_ligand.sdf")

    def create_all_folders(self):
        os.makedirs(osp.join(self.polar_root, "ligands", "neutral"), exist_ok=True)
        os.makedirs(osp.join(self.polar_root, "ligands", "protonated"), exist_ok=True)
        os.makedirs(self.protein_pdbqt_root, exist_ok=True)
        os.makedirs(self.ligand_pdbqt_root, exist_ok=True)
        os.makedirs(self.linf9_opt_root, exist_ok=True)
        os.makedirs(self.martini_root, exist_ok=True)

    @property
    def info_df(self):
        if self._info_df is None:
            self._info_df = pd.read_csv(self.info_csv, skiprows=6, header=None, sep="\s+",
                names=["pdb", "resolution", "year", "pKd", "pKd_src", "IGNORE", "ref", "ligand_name"])
        return self._info_df
    
    @property
    def linf9_df(self) -> pd.DataFrame:
        if self._linf9_df is None:
            self._linf9_df = pd.read_csv(self.linf9_csv)
        return self._linf9_df
    
    def __len__(self):
        return self.info_df.shape[0]

    @property
    def pdb2pkd(self) -> Dict[str, float]:
        if self._pdb2pkd is None:
            res = self.info_df[["pdb", "pKd"]].set_index("pdb").to_dict()["pKd"]
            self._pdb2pkd = res
        return self._pdb2pkd

    @property
    def iter_protein_pdbqt(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.protein_pdbqt_root, f"{pdb}.pdbqt")
        for pdb in self.others_pdbs:
            yield osp.join(self.protein_pdbqt_root, f"{pdb}.pdbqt")

    @property
    def iter_ligand_pdbqt(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.ligand_pdbqt_root, f"{pdb}.pdbqt")
        for pdb in self.others_pdbs:
            yield osp.join(self.ligand_pdbqt_root, f"{pdb}.pdbqt")

    @property
    def iter_linf9_opt_ligands(self):
        for pdb in self.refined_pdbs:
            yield osp.join(self.linf9_opt_root, f"{pdb}.pdb")
        for pdb in self.others_pdbs:
            yield osp.join(self.linf9_opt_root, f"{pdb}.pdb")

    def pdb2linf9_opt_lig(self, pdb: str):
        return osp.join(self.linf9_opt_root, f"{pdb}.pdb")

    @property
    def ds_size(self):
        if self._ds_size is None:
            self._ds_size = len(self.refined_pdbs) + len(self.others_pdbs)
        return self._ds_size

    @property
    def refined_pdbs(self) -> Set[str]:
        if self._refined_pdbs is None:
            folders = glob(osp.join(self.refined_set_root, "*"))
            pdbs = [osp.basename(f) for f in folders]
            pdbs = set(pdbs)
            pdbs.remove("index")
            pdbs.remove("readme")
            self._refined_pdbs = pdbs
        return self._refined_pdbs

    @property
    def others_pdbs(self) -> Set[str]:
        if self._others_pdbs is None:
            folders = glob(osp.join(self.others_root, "*"))
            pdbs = [osp.basename(f) for f in folders]
            pdbs = set(pdbs)
            pdbs.remove("index")
            pdbs.remove("readme")
            self._others_pdbs = pdbs
        return self._others_pdbs
    
    def pdb2obabel_lig(self, pdb: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.Obabel", f"{pdb}.sdf")
    
    @lazy_property
    def obabel_success_yaml(self) -> str:
        return osp.join(self.phys_infuse_root, "info", "obabel.success.yaml")
    
    @lazy_property
    def obabel_pdbs(self) -> Set[str]:
        # return a list of PDBs that are cleaned by OpenBabel to Kekulize atoms
        with open(self.obabel_success_yaml) as f:
            pdbs: List[str] = yaml.safe_load(f)
        return set(pdbs)
        
    def pdb2lig_allh_cleaned(self, pdb: str) -> str:
        # Return ligand file (all hydrogen) that can be parsed by RDKit successfuly
        # Due to kekulize problem, about 3000 ligands in the original data set cannot be parsed
        # These ligands are cleaned by OpenBabel.
        if pdb in self.obabel_pdbs:
            return self.pdb2obabel_lig(pdb)
        return self.pdb2neutral_lig(pdb)
    
    def pdb2mmff_allh(self, pdb: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.MMFF", f"{pdb}.sdf")
    

SAVE_ROOT = "/scratch/sx801/data/im_datasets/processed"
def gen_pdb2020_split(ds_name):
    set_num = 1
    rtm_val = f"/scratch/sx801/scripts/RTMScore/old/data/val_set{set_num}"

    val_set_pdbs = pd.read_csv(rtm_val, header=None).values
    val_set_pdbs = set(val_set_pdbs.reshape(-1).tolist())

    train_index = []
    val_index = []
    d = torch.load(f"/scratch/sx801/data/im_datasets/processed/{ds_name}.pyg")
    casf_pdbs = [osp.basename(f) for f in glob("/vast/sx801/geometries/CASF-2016-cyang/coreset/????")]
    casf_pdbs = set(casf_pdbs)
    assert len(casf_pdbs) == 285, casf_pdbs
    for i, f in enumerate(d[0].ligand_file):
        this_pdb = osp.basename(f[0]).split("_")[0].split(".")[0]
        if this_pdb in casf_pdbs:
            print(this_pdb, " is removed because it is in the CASF dataset.")
            continue
        if this_pdb in val_set_pdbs:
            val_index.append(i)
        else:
            train_index.append(i)
    torch.save({"train_index": train_index, "val_index": val_index, "test_index": None},
         osp.join(SAVE_ROOT, ds_name+f".rtmsplit{set_num}.pth"))
    print("Training size: ", len(train_index))
    print("Validation size: ", len(val_index))

def gen_pdb2020_split_filtered(ds_name):
    with open(osp.join(SAVE_ROOT, "PBind2020OG.split.covbinder.res2.5.testsets.poslinf9.metal.yaml")) as f:
        split = yaml.safe_load(f)
    # update 12/14/2023 we found some ligand structures with duplicate atoms, 
    # they are also going to be removed
    with open("/vast/sx801/geometries/PL_physics_infusion/PDBBind2020_OG/info/duplicate_atom_pdbs.json") as f:
        duplicate_atom_pdbs: List[str] = json.load(f)
    duplicate_atom_pdbs: Set[str] = set(duplicate_atom_pdbs)
    train_set_pdbs = set(split["train"])
    val_set_pdbs = set(split["val"])

    train_index = []
    val_index = []
    d = torch.load(f"/scratch/sx801/data/im_datasets/processed/{ds_name}.pyg")
    for i, this_pdb in enumerate(d[0].pdb):
        if this_pdb in duplicate_atom_pdbs:
            continue
        if this_pdb in val_set_pdbs:
            val_index.append(i)
        elif this_pdb in train_set_pdbs:
            train_index.append(i)
    torch.save({"train_index": train_index, "val_index": val_index, "test_index": None},
         osp.join(SAVE_ROOT, ds_name+f".covbinder.res2.5.testsets.poslinf9.metal.pth"))
    print("Training size: ", len(train_index))
    print("Validation size: ", len(val_index))

def main():
    # check numbers
    pdb2020_root = "/scratch/sx801/temp"
    ds_reader = PDB2020DSReader(pdb2020_root)
    print(ds_reader.info_df)
    print(ds_reader.pdb2pkd)
    print(len(ds_reader.refined_pdbs))
    print(len(ds_reader.others_pdbs))

    coreset_pdbs = [osp.basename(f) for f in glob("/vast/sx801/CASF-2016-cyang/coreset/*")]
    coreset_pdbs = set(coreset_pdbs)
    print("---")
    print(len(coreset_pdbs.intersection(ds_reader.refined_pdbs)))
    print(len(coreset_pdbs.intersection(ds_reader.others_pdbs)))

def main1():
    # testing yield
    pdb2020_root = "/scratch/sx801/temp"
    ds_reader = PDB2020DSReader(pdb2020_root)
    roots = [f for f in ds_reader.iter_pdb_roots()]
    print(len(roots))

if __name__ == "__main__":
    main()
