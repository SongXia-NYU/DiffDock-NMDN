import json
from typing import Set, Tuple, List
import urllib
import urllib.request
import io
import Bio
import Bio.PDB
import os
import os.path as osp

# Adapated from the dataset curation code of CovBinderInPDB:
# Guo, X.-K. & Zhang, Y. CovBinderInPDB: A Structure-Based Covalent Binder Database. J Chem Inf Model 62, 6057–6068 (2022).
class CovalentBondFilter:
    def __init__(self, cif_root: str, debug=False, criterions: Set[str]=None) -> None:
        self.cif_root = cif_root
        self.debug = debug

        # set up criterions
        all_criterions = set(["acceptable_nucleophile", "invalid_binder", "acceptable_element"])
        if criterions is None:
            criterions = all_criterions
        assert len(criterions.difference(all_criterions)) == 0, criterions
        self._do_filter_nucleophile = "acceptable_nucleophile" in criterions
        self._do_filter_binder = "invalid_binder" in criterions
        self._do_filter_element = "acceptable_element" in criterions

        self.const_file = osp.join(osp.dirname(__file__), "covalent_bond_consts.json")

        self._consts = None
        self._acceptable_nucleophile = None
        self._invalid_binder = None
        self._acceptable_element = None

    @property
    def acceptable_nucleophile(self):
        # We are not considering terminal amine although it could be the target of covalent binder. 
        if self._acceptable_nucleophile is None:
            self._acceptable_nucleophile = [tuple(i) for i in self.consts["acceptable_nucleophile"]]
        return self._acceptable_nucleophile

    @property
    def invalid_binder(self):
        # some components are not binders (metal-containing ligand? solvent? ion?)
        if self._invalid_binder is None:
            self._invalid_binder = self.consts["invalid_binder"]
        return self._invalid_binder

    @property
    def acceptable_element(self):
        if self._acceptable_element is None:
            self._acceptable_element = set(self.consts["acceptable_element"])
        return self._acceptable_element

    @property
    def consts(self):
        if self._consts is not None:
            return self._consts
        
        with open(self.const_file) as f:
            self._consts = json.load(f)
        return self._consts

    def retrive_blc(self, pdb_id: str) -> dict:
        pdb_id = pdb_id.lower()
        # no saving CIFs
        if self.cif_root is None:
            r = urllib.request.urlopen(f'http://files.rcsb.org/download/{pdb_id}.cif')
            f = io.StringIO(r.read().decode())
            if self.debug: print(f'We fetched {pdb_id}.cif from RCSB PDB')
            blc = Bio.PDB.MMCIF2Dict.MMCIF2Dict(f)
            return blc
        
        ckpt_prot = osp.join(self.cif_root, f"{pdb_id}.cif")
        if not osp.exists(ckpt_prot):
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.cif', ckpt_prot)
            if self.debug: print(f'We fetched {pdb_id}.cif from RCSB PDB')
        blc = Bio.PDB.MMCIF2Dict.MMCIF2Dict(ckpt_prot)
        if self.debug: print(f'We have local copy of {pdb_id}.cif')
        return blc
    
    def covale_col(self, blc: dict, conn_idx: int) -> list:
        col = []
        for j, label_entity_id in enumerate(blc['_atom_site.label_entity_id']):
            if  blc["_atom_site.label_asym_id"    ][j] == blc["_struct_conn.ptnr1_label_asym_id"     ][conn_idx] and \
                blc["_atom_site.auth_asym_id"     ][j] == blc["_struct_conn.ptnr1_auth_asym_id"      ][conn_idx] and \
                blc["_atom_site.label_comp_id"    ][j] == blc["_struct_conn.ptnr1_label_comp_id"     ][conn_idx] and \
                blc["_atom_site.auth_comp_id"     ][j] == blc["_struct_conn.ptnr1_auth_comp_id"      ][conn_idx] and \
                blc["_atom_site.label_seq_id"     ][j] == blc["_struct_conn.ptnr1_label_seq_id"      ][conn_idx] and \
                blc["_atom_site.auth_seq_id"      ][j] == blc["_struct_conn.ptnr1_auth_seq_id"       ][conn_idx] and \
                blc["_atom_site.pdbx_PDB_ins_code"][j] == blc["_struct_conn.pdbx_ptnr1_PDB_ins_code" ][conn_idx]:
                for k, id_ in enumerate(blc["_entity.id"]):
                    if label_entity_id == id_:
                        col.append(blc["_entry.id"][0])                                    #0
                        col.append(blc["_struct_conn.ptnr1_label_asym_id"     ][conn_idx]) #1
                        col.append(blc["_struct_conn.ptnr1_auth_asym_id"      ][conn_idx]) #2
                        col.append(blc["_struct_conn.ptnr1_label_comp_id"     ][conn_idx]) #3
                        col.append(blc["_struct_conn.ptnr1_auth_comp_id"      ][conn_idx]) #4
                        col.append(blc["_struct_conn.ptnr1_label_seq_id"      ][conn_idx]) #5
                        col.append(blc["_struct_conn.ptnr1_auth_seq_id"       ][conn_idx]) #6
                        col.append(blc["_struct_conn.pdbx_ptnr1_PDB_ins_code" ][conn_idx]) #7
                        col.append(blc["_struct_conn.ptnr1_label_atom_id"     ][conn_idx]) #8
                        col.append(blc["_struct_conn.pdbx_ptnr1_label_alt_id" ][conn_idx]) #9
                        col.append(blc["_entity.type"][k])                                 #10
                        break
                break
        for j, label_entity_id in enumerate(blc['_atom_site.label_entity_id']):
            if  blc["_atom_site.label_asym_id"    ][j] == blc["_struct_conn.ptnr2_label_asym_id"     ][conn_idx] and \
                blc["_atom_site.auth_asym_id"     ][j] == blc["_struct_conn.ptnr2_auth_asym_id"      ][conn_idx] and \
                blc["_atom_site.label_comp_id"    ][j] == blc["_struct_conn.ptnr2_label_comp_id"     ][conn_idx] and \
                blc["_atom_site.auth_comp_id"     ][j] == blc["_struct_conn.ptnr2_auth_comp_id"      ][conn_idx] and \
                blc["_atom_site.label_seq_id"     ][j] == blc["_struct_conn.ptnr2_label_seq_id"      ][conn_idx] and \
                blc["_atom_site.auth_seq_id"      ][j] == blc["_struct_conn.ptnr2_auth_seq_id"       ][conn_idx] and \
                blc["_atom_site.pdbx_PDB_ins_code"][j] == blc["_struct_conn.pdbx_ptnr2_PDB_ins_code" ][conn_idx]:
                for k, id_ in enumerate(blc["_entity.id"]):
                    if label_entity_id == id_: 
                        col.append(blc["_struct_conn.ptnr2_label_asym_id"     ][conn_idx]) #11
                        col.append(blc["_struct_conn.ptnr2_auth_asym_id"      ][conn_idx]) #12
                        col.append(blc["_struct_conn.ptnr2_label_comp_id"     ][conn_idx]) #13
                        col.append(blc["_struct_conn.ptnr2_auth_comp_id"      ][conn_idx]) #14
                        col.append(blc["_struct_conn.ptnr2_label_seq_id"      ][conn_idx]) #15
                        col.append(blc["_struct_conn.ptnr2_auth_seq_id"       ][conn_idx]) #16
                        col.append(blc["_struct_conn.pdbx_ptnr2_PDB_ins_code" ][conn_idx]) #17
                        col.append(blc["_struct_conn.ptnr2_label_atom_id"     ][conn_idx]) #18
                        col.append(blc["_struct_conn.pdbx_ptnr2_label_alt_id" ][conn_idx]) #19
                        col.append(blc["_entity.type"][k])                                 #20
                        break 
                break
        return col
    
    def filter_nucleophile(self, col: list) -> Tuple[list, bool]:
        if not self._do_filter_nucleophile:
            return col, False
        
        # True: filtered, False: passed
        if   (col[4],  col[8] ) in self.acceptable_nucleophile and col[10] == 'polymer': 
            pass
        elif (col[14], col[18]) in self.acceptable_nucleophile and col[20] == 'polymer': 
            col = [col[0]] + col[11:] + col[1:11] # Reconstruct the list
        else: 
            if self.debug: print(f"Invalid covalent bond record: {','.join(col)}")
            return col, True # skip invalid covalent modification
        return col, False
    
    def filter_binder(self, col: list) -> bool:
        if not self._do_filter_binder:
            return False
        
        # True: filtered, False: passed
        if col[14] in self.invalid_binder: 
            if self.debug: print(f'The binder {col[14]} is invalid because of {self.invalid_binder[col[14]]}')
            return True
        return False

    def filter_elements(self, binder_elements: Set[int]) -> bool:
        if not self._do_filter_element:
            return False
        
        # True: filtered, False: passed
        if binder_elements.difference(self.acceptable_element): # We screen the binder when it is in the adduct. If it has metal element, we skip the bond record.
            if self.debug: print(f'The binder in this record is invalid because it contains {binder_elements.difference(self.acceptable_element)}')
            return True
        return False

    def get_covalent_bond_record(self, pdb_id: str):
        blc = self.retrive_blc(pdb_id)
        if "_struct_conn.id" not in blc: return [] # struct_conn_IS_NOT_FOUND 

        cbr = []
        for conn_idx, conn_type_id in enumerate(blc["_struct_conn.conn_type_id"]):
            if conn_type_id != 'covale':
                continue

            # col example:
            #    0 1 2   3   4   5   6 7 8 9      10 1 2   3   4   5   6 7 8 9      20    
            # 2XAZ,A,A,LEU,LEU,729,729,?,C,?,polymer,A,A,NIY,NIY,730,730,?,N,?,polymer
            col = self.covale_col(blc, conn_idx)
            
            col, filtered = self.filter_nucleophile(col)
            if filtered: 
                continue # skip invalid covalent modification

            if self.filter_binder(col):
                continue

            binder_elements = set()
            for i, type_symbol in enumerate(blc['_atom_site.type_symbol']):  
                if blc["_atom_site.auth_asym_id"][i]==col[12] and blc["_atom_site.auth_comp_id"][i]==col[14]: 
                    binder_elements.add(type_symbol) 
            if self.filter_elements(binder_elements):
                continue
            
            cbr.append(','.join(col))
        return cbr


def benchmark():
    targets = ["1bio", "1a30", "7c7p", "1h8y", "1fdz", "3vpk", "6skq", "2o9a", "4jmc"]
    with open("covalent_bond_benchmark.json") as f:
        expected = json.load(f)
    filter = CovalentBondFilter("./temp", False)
    for target in targets:
        cbr = filter.get_covalent_bond_record(target)
        if cbr == expected[target]:
            print(f"PASS: {target}")
            continue

        print(f"ERROR: {target} did not pass!")
        print("Expected: ", expected[target])
        print("Predicted: ", cbr)
        print("--"*20)

def gen_benchmark():
    exit(0)
    targets = ["1bio", "1a30", "7c7p", "1h8y", "1fdz", "3vpk", "6skq", "2o9a", "4jmc"]
    out = {}
    for target in targets:
        out[target] = get_covalent_bond_record(target)
    with open("covalent_bond_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    benchmark()
