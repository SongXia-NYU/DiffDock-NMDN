import os.path as osp
from glob import glob
from typing import List, Set, Tuple, Optional

from geometry_processors.lazy_property import lazy_property

class ProtProtReader:
    def __init__(self, droot: str) -> None:
        self.droot: str = droot
        self.polarh_root: str = osp.join(droot, "prot_polarH")
        self.test_root: str = osp.join(droot, "AF_models_separated_chains_polarH")

    def pdb2pairs(self, pdb: str) -> Tuple[Optional[str], Optional[str]]:
        chain1: str = osp.join(self.polarh_root, f"{pdb}_chain_1.pdb")
        if not osp.exists(chain1):
            return None, None
        
        chain2: str = osp.join(self.polarh_root, f"{pdb}_chain_2.pdb")
        if not osp.exists(chain2):
            return None, None
        return chain1, chain2

    @lazy_property
    def pdbs(self) -> Set[str]:
        valid_pdbs_file: str = osp.join(self.droot, "A07_01_valid_files_after_polarH.txt")
        with open(valid_pdbs_file) as f:
            lines: List[str] = f.readlines()
        pdbs: Set[str] = set()
        for line in lines:
            this_pdb: str = line.strip().split("_")[0]
            assert len(this_pdb) == 4, this_pdb
            pdbs.add(this_pdb)
        return pdbs
    
    @lazy_property
    def test_pdbs(self) -> Set[str]:
        test_pdb_roots: List[str] = glob(osp.join(self.test_root, "????"))
        test_pdbs: List[str] = [osp.basename(test_pdb_root) for test_pdb_root in test_pdb_roots]
        return set(test_pdbs)
    
    def pdb2test_pairs(self, pdb: str) -> List[Tuple[str, str]]:
        this_root: str = osp.join(self.test_root, pdb)
        pdb_files: Set[str] = set([osp.basename(f) for f in glob(osp.join(this_root, "ranked_*_chain_?.pdb"))])
        ranks: Set[str] = set()
        for pdb_f in pdb_files:
            ranks.add(pdb_f.split("_")[1])

        test_pairs: List[Tuple[str, str]] = []
        for rank in ranks:
            chain1_pdb = f"ranked_{rank}_chain_1.pdb"
            chain2_pdb = f"ranked_{rank}_chain_2.pdb"
            if chain1_pdb not in pdb_files or chain2_pdb not in pdb_files:
                continue
            test_pairs.append((osp.join(this_root, chain1_pdb), osp.join(this_root, chain2_pdb)))
        return test_pairs
