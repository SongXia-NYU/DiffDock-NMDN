from collections import defaultdict
import os
import os.path as osp
from glob import glob
from typing import List
from rdkit.Chem import MolFromMol2Block, MolFromMol2File, Mol, SDMolSupplier

import pandas as pd

class CASF2016Reader:
    def __init__(self, droot: str, phys_infuse_root: str = None) -> None:
        self.droot = droot
        self.phys_infuse_root = phys_infuse_root
        # core set for scoring power
        self.coreset_root = osp.join(droot, "coreset")
        self.coreset_amend = osp.join(droot, "coreset_amend")
        self.coreset_renum_root = osp.join(droot, "coreset_fixed")
        self.coreset_polar_root = osp.join(droot, "coreset_polarH")
        self.coreset_noh_root = osp.join(droot, "coreset_noh")

        # decoys for docking power
        self.dock_root = osp.join(droot, "decoys_docking")
        self.dock_polar_root = osp.join(droot, "decoys_docking_polarH")
        self.dock_noh_root = osp.join(droot, "decoys_docking_noH")

        # decoys for screening power
        self.screen_root = osp.join(droot, "decoys_screening")
        self.screen_polar_root = osp.join(droot, "decoys_screening_polarH")
        self.screen_noh_root = osp.join(droot, "decoys_screening_noH")

        self._pdbs = None
        self._screen_pdbs = None
        self._dock_polarh_mapper = None
        self._label_df = None 

    @property
    def pdbs(self) -> List[str]:
        if self._pdbs is None:
            self._pdbs = [osp.basename(p) for p in glob(f"{self.coreset_root}/*")]
            self._pdbs.sort()
        return self._pdbs
    
    @property
    def screen_pdbs(self) -> List[str]:
        # the 57 PDB ids of the targets
        if self._screen_pdbs is None:
            self._screen_pdbs = [osp.basename(p) for p in glob(f"{self.screen_root}/*")]
            self._screen_pdbs.sort()
        return self._screen_pdbs
    
    def pdb2prot_renumh(self, pdb: str) -> str:
        return osp.join(self.coreset_renum_root, pdb, f"{pdb}_protein.renum.pdb")
    
    def pdb2prot_polarh(self, pdb: str) -> str:
        return osp.join(self.coreset_polar_root, pdb, f"{pdb}_protein.polar.pdb")
    
    def pdb2lig_core_polarh(self, pdb: str) -> str:
        return osp.join(self.coreset_polar_root, pdb, f"{pdb}_ligand.polar.mol2")
    
    def pdb2lig_core_sdf(self, pdb: str) -> str:
        return osp.join(self.coreset_root, pdb, f"{pdb}_ligand.sdf")
    
    def pdb2lig_core_mol2(self, pdb: str) -> str:
        return osp.join(self.coreset_root, pdb, f"{pdb}_ligand.mol2")
    
    # not yet used
    def pdb2lig_core_obabel_sdf(self, pdb: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.Obabel", f"{pdb}_ligand.sdf")
    
    # not yet used
    def pdb2lig_core_mmff_sdf(self, pdb: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.MMFF", f"{pdb}_ligand.sdf")
    
    def pdb2lig_amend(self, pdb: str) -> str:
        return osp.join(self.coreset_amend, f"{pdb}_ligand.sdf")

    def pdb2prot_noh(self, pdb: str) -> str:
        return osp.join(self.coreset_noh_root, pdb, f"{pdb}_protein.noh.pdb")
    
    def pdb2lig_core_noh(self, pdb: str) -> str:
        return osp.join(self.coreset_noh_root, pdb, f"{pdb}_ligand.noh.mol2")
    
    def pdb2dock_ligs(self, pdb: str) -> str:
        return osp.join(self.dock_root, f"{pdb}_decoys.mol2")
    
    def pdb2dock_fix_ligs(self, pdb: str) -> str:
        return osp.join(self.dock_root, f"{pdb}_decoys.fix.sdf")
    
    def pdb2dock_mmff_ligs(self, pdb: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.Docking.MMFF", f"{pdb}_decoys.mmff.sdf")
    
    def pdb2dock_polarh_ligs(self, pdb: str) -> List[str]:
        # return a list of mol2 files (around 100 files)
        return self.dock_polarh_mapper[pdb]
    
    def pdb2dock_noh_ligs(self, pdb: str) -> str:
        return osp.join(self.dock_noh_root, f"{pdb}_decoys.noh.mol2")
    
    def pdb2screen_allh_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        return osp.join(self.screen_root, pdb_tgt, f"{pdb_tgt}_{pdb_lig}.mol2")
    
    def pdb2screen_allh_obabel_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.Screening.Obabel", pdb_tgt, f"{pdb_tgt}_{pdb_lig}.sdf")
    
    def pdb2screen_allh_fix_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        obabel_sdf = self.pdb2screen_allh_obabel_ligs(pdb_tgt, pdb_lig)
        if osp.exists(obabel_sdf):
            return obabel_sdf
        return self.pdb2screen_allh_ligs(pdb_tgt, pdb_lig)
    
    def pdb2screen_allh_mmff_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        return osp.join(self.phys_infuse_root, "Ligands.AllH.Screening.MMFF", pdb_tgt, f"{pdb_tgt}_{pdb_lig}.mmff.sdf")
        
    def pdb2screen_polarh_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        return osp.join(self.screen_polar_root, pdb_tgt, f"{pdb_tgt}_{pdb_lig}.polar.mol2")
    
    def pdb2screen_noh_ligs(self, pdb_tgt: str, pdb_lig: str) -> str:
        return osp.join(self.screen_noh_root, pdb_tgt, f"{pdb_tgt}_{pdb_lig}.noh.mol2")
    
    def pdb2pkd(self, pdb: str):
        return self.label_df.loc[pdb]["pKd"].item()
    
    @property
    def dock_polarh_mapper(self):
        if self._dock_polarh_mapper is not None:
            return self._dock_polarh_mapper
        
        dock_mapper = defaultdict(lambda: [])
        for f in glob(osp.join(self.dock_polar_root, "????_single*.polar.mol2")):
            pdb = osp.basename(f).split("_")[0]
            dock_mapper[pdb].append(f)
        self._dock_polarh_mapper = dock_mapper
        return self._dock_polarh_mapper
    
    @property
    def label_df(self):
        if self._label_df is None:
            label_df = pd.read_csv(osp.join(self.droot, "CASF-2016.csv")).set_index("pdb")
            self._label_df = label_df
        return self._label_df
    
def polar_mol2_to_info(prot_pdb: str, lig_mol2: str, pdb: str, **kwargs):
    from geometry_processors.pl_dataset.csv2input_list import MPInfo
    # special treatment to polarH since I forget to save the ligand_name which turned out to be important
    mol = MolFromMol2File(lig_mol2, removeHs=False, sanitize=False, cleanupSubstructures=False)
    mol2_forname = lig_mol2.replace(".polar.mol2", ".mol2").replace("coreset_polarH", "coreset")
    molforname = MolFromMol2File(mol2_forname, removeHs=False, sanitize=False, cleanupSubstructures=False)
    name = molforname.GetProp("_Name")
    this_info = MPInfo(protein_pdb=prot_pdb, mol=mol, ligand_id=name, pdb=pdb, **kwargs)
    return this_info

def mol2_to_info_list(prot_pdb: str, lig_mol2: str, pdb: str, mol_read_args:dict=None, **kwargs):
    from geometry_processors.pl_dataset.csv2input_list import MPInfo
    # lig_mol2 contains multiple molecules
    SEP = "@<TRIPOS>MOLECULE"
    with open(lig_mol2) as f:
        txt = f.read()
    mol_blocks = [SEP+s for s in txt.split(SEP)[1:]]
    info_list = []
    if mol_read_args is None:
        mol_read_args = {"removeHs": False, "sanitize": False, "cleanupSubstructures": False}
    for block in mol_blocks:
        mol = MolFromMol2Block(block, **mol_read_args)
        try:
            name = mol.GetProp("_Name")
        except KeyError as e:
            print(block)
            print("--"*10)
            raise e

        this_info = MPInfo(protein_pdb=prot_pdb, mol=mol, ligand_id=name, pdb=pdb, **kwargs)
        info_list.append(this_info)
    return info_list

def mol_file2mol_list(mol_file: str) -> List[Mol]:
    if mol_file.endswith(".mol2"):
        info_list = mol2_to_info_list(None, mol_file, None, {})
        return [info.mol for info in info_list]
    
    assert mol_file.endswith(".sdf")
    return [mol for mol in SDMolSupplier(mol_file)]

