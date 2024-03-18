"""
Map the atom index to Martini index for variety of usages.
"""
import torch
from prody import parsePDB

class MartiniMapper:
    def __init__(self, atom_pdb, cg_pdb) -> None:
        self.atom_pdb = atom_pdb
        self.cg_pdb = cg_pdb

        self.atom_reader = parsePDB(self.atom_pdb)
        self.cg_reader = parsePDB(self.cg_pdb)

    def get_batch_index(self):
        # return a "batch index" style mapping from atom-level to martini-level
        # for the atoms that does not belong any bead (i.e., non-polar hydrogens), -1 is assigned
        batch_index = torch.zeros(self.atom_reader.numAtoms()).long().fill_(-1)
        for atom_res, cg_res in zip(self.atom_reader.iterResidues(), self.cg_reader.iterResidues()):
            this_res_name = atom_res.getResnames()[0]
            assert this_res_name == cg_res.getResnames()[0]
            this_martini_mapper = _MARTINI_MAPPER[this_res_name]
            for atoms, bead_name in zip(this_martini_mapper, _BEAD_NAMES):
                select_cg = cg_res.select(f"name {bead_name}")
                select_atoms = atom_res.select(f"name {' '.join(atoms)}")
                if select_atoms is None:
                    continue
                bead_idx = select_cg.getIndices()
                assert len(bead_idx) == 1, bead_idx
                bead_idx = bead_idx[0]
                for atom_idx in select_atoms.getIndices():
                    batch_index[atom_idx] = bead_idx
        return batch_index

def get_martini_mapping():
    # The mapping used in Martini Coarse Graining for proteins
    # It is adapted from the official implementation: https://github.com/cgmartini/martinize.py/blob/master/MAP.py

    # Split each argument in a list
    def nsplit(*x):
        return [i.split() for i in x]

    bb = "N CA C O H H1 H2 H3 O1 O2"
    # This is the mapping dictionary
    # For each residue it returns a list, each element of which
    # lists the atom names to be mapped to the corresponding bead.
    # The order should be the standard order of the coarse grained
    # beads for the residue. Only atom names matching with those
    # present in the list of atoms for the residue will be used
    # to determine the bead position. This adds flexibility to the
    # approach, as a single definition can be used for different
    # states of a residue (e.g., GLU/GLUH).
    # For convenience, the list can be specified as a set of strings,
    # converted into a list of lists by 'nsplit' defined above.
    mapping = {
        "ALA":  nsplit(bb + " CB"),
        "CYS":  nsplit(bb, "CB SG"),
        "ASP":  nsplit(bb, "CB CG OD1 OD2"),
        "GLU":  nsplit(bb, "CB CG CD OE1 OE2"),
        "PHE":  nsplit(bb, "CB CG CD1 HD1", "CD2 HD2 CE2 HE2", "CE1 HE1 CZ HZ"),
        "GLY":  nsplit(bb),
        "HIS":  nsplit(bb, "CB CG", "CD2 HD2 NE2 HE2", "ND1 HD1 CE1 HE1"),
        "HIH":  nsplit(bb, "CB CG", "CD2 HD2 NE2 HE2", "ND1 HD1 CE1 HE1"),     # Charged Histidine.
        "ILE":  nsplit(bb, "CB CG1 CG2 CD CD1"),
        "LYS":  nsplit(bb, "CB CG CD", "CE NZ HZ1 HZ2 HZ3"),
        "LEU":  nsplit(bb, "CB CG CD1 CD2"),
        "MET":  nsplit(bb, "CB CG SD CE"),
        "ASN":  nsplit(bb, "CB CG ND1 ND2 OD1 OD2 HD11 HD12 HD21 HD22"),
        "PRO":  nsplit(bb, "CB CG CD"),
        "HYP":  nsplit(bb, "CB CG CD OD"),
        "GLN":  nsplit(bb, "CB CG CD OE1 OE2 NE1 NE2 HE11 HE12 HE21 HE22"),
        "ARG":  nsplit(bb, "CB CG CD", "NE HE CZ NH1 NH2 HH11 HH12 HH21 HH22"),
        "SER":  nsplit(bb, "CB OG HG"),
        "THR":  nsplit(bb, "CB OG1 HG1 CG2"),
        "VAL":  nsplit(bb, "CB CG1 CG2"),
        "TRP":  nsplit(bb, "CB CG CD2", "CD1 HD1 NE1 HE1 CE2", "CE3 HE3 CZ3 HZ3", "CZ2 HZ2 CH2 HH2"),
        "TYR":  nsplit(bb, "CB CG CD1 HD1", "CD2 HD2 CE2 HE2", "CE1 HE1 CZ OH HH")
        }
    return mapping

def get_bead_names():
    # Generic names for side chain beads
    residue_bead_names = "BB SC1 SC2 SC3 SC4".split()
    return residue_bead_names

_MARTINI_MAPPER = get_martini_mapping()
_BEAD_NAMES = get_bead_names()

if __name__ == "__main__":
    mapper = MartiniMapper("/scratch/sx801/temp/RenumPDBs/1a07.renum.pdb", "/scratch/sx801/temp/Martini/pdb/1a07.renum.martini.pdb")
    mapper.get_batch_index()

