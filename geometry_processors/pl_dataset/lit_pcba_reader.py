import json
import os
import os.path as osp
from glob import glob
from typing import List, Tuple
from tqdm import tqdm
from geometry_processors.lazy_property import lazy_property
from geometry_processors.pl_dataset.prot_utils import pdb2seq

# the PDB_ID used for DiffDock docking
# designed to be consistent with GenScore Paper: https://pubs.rsc.org/en/content/articlelanding/2023/SC/D3SC02044D
pdb_getter = {
    "ADRB2": "4ldo", "ALDH1": "5l2m", "ESR1_ago": "2p15",
    "ESR1_ant": "2iok", "FEN1": "5fv7", "GBA": "2v3d",
    "IDH1": "4umx", "KAT2A": "5h86", "MAPK1": "4zzn",
    "MTORC1": "4dri", "OPRK1": "6b73", "PKM2": "4jpg",
    "PPARG": "5y2t", "TP53": "3zme", "VDR": "3a2i"
}

TARGETS = {"ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A", "MAPK1", 
"MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"}

class LIT_PCBA_Reader:
    def __init__(self, pcba_root: str, target: str) -> None:
        self.pcba_root = pcba_root
        # target example: ESR1_ant, GBA
        self.target = target
        self.droot = osp.join(pcba_root, target)
        self.fl_json = osp.join(self.droot, "info", "file_handle.json")

        self._fl = None

    @property
    def polar_prot2seq_info(self):
        info = {}
        for pdb in self.prot_pdbs():
            prot = self.pdb2polar_prot(pdb)
            info[pdb] = pdb2seq(prot)
        return info

    @lazy_property
    def fl(self) -> List[str]:     
        # cache on file
        if osp.exists(self.fl_json):
            with open(self.fl_json) as f:
                self._fl = json.load(f)
            return self._fl
        
        os.makedirs(osp.dirname(self.fl_json), exist_ok=True)
        lig_files = glob(osp.join(self.droot, "pose", "*.pdb"))
        fl = [self.lig_file2pdb(f)[-1] for f in lig_files]
        with open(self.fl_json, "w") as f:
            json.dump(fl, f, indent=2)
        return fl

    def polar_ligs(self) -> List[str]:
        return [osp.join(self.droot, "pose", f"{fl}.pdb") for fl in self.fl]
    
    def prot_pdbs(self) -> List[str]:
        prot_files = glob(osp.join(self.droot, "????_protein.mol2"))
        pdb_list = [osp.basename(p).split(".")[0].split("_")[0] for p in prot_files]
        return pdb_list
    
    def pdb2raw_prot_mol2(self, pdb: str) -> str:
        return osp.join(self.droot, f"{pdb}_protein.mol2")
    
    def pdb2raw_prot(self, pdb: str) -> str:
        return osp.join(self.droot, "MartiniPreproc", f"{pdb}.raw.pdb")
    
    def pdb2renumh_prot(self, pdb: str) -> str:
        return osp.join(self.droot, "MartiniPreproc", f"{pdb}.renumH.pdb")
    
    def pdb2polar_prot(self, pdb: str) -> str:
        return osp.join(self.droot, "PolarH", f"{pdb}.polar.pdb")
    
    @staticmethod
    def lig_file2pdb(lig_file: str) -> Tuple[str, str]:
        file_handle = osp.basename(lig_file).split(".pdb")[0]
        pdb_id = file_handle.split("_")[-1]
        return pdb_id, file_handle


class LIT_PCBA_DiffDockReader(LIT_PCBA_Reader):
    """
    LIT-PCBA dataset, but ligand geometries are generated using DiffDock instead of Lin_F9.
    """
    def __init__(self, pcba_root: str, target: str) -> None:
        super().__init__(pcba_root, target)
        self.fl_json = osp.join(self.droot, "info", "file_handle_diffdock.json")

    @lazy_property
    def fl(self) -> List[str]:
        if osp.exists(self.fl_json):
            return super().fl
        
        lig_files = glob(osp.join(self.droot, "DiffDock_D021_D022", "polar_h", "*", "rank*_confidence-*.sdf"))
        fl = []
        for lig_file in tqdm(lig_files):
            sdf_name: str = osp.basename(lig_file)
            pl_pair: str = osp.basename(osp.dirname(lig_file))
            this_fl = f"{pl_pair}/{sdf_name}"
            fl.append(this_fl)

        with open(self.fl_json, "w") as f:
            json.dump(fl, f, indent=2)
        return fl
    
    def polar_ligs(self) -> List[str]:
        return [self.fl2polar_lig_sdf(fl) for fl in self.fl]
    
    def fl2polar_lig_sdf(self, fl: str):
        return osp.join(self.droot, "DiffDock_D021_D022", "polar_h", *fl.split("/"))
    
    @staticmethod
    def lig_file2pdb(lig_file: str) -> Tuple[str, str]:
        pl_pair = osp.basename(osp.dirname(lig_file))
        pdb_id = pl_pair.split("_")[-1]
        return pdb_id, pl_pair
