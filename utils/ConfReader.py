import os.path as osp

import json
import ase
import numpy as np
import torch
from prody.proteins import parsePDB
from rdkit.Chem import MolFromPDBFile, AddHs, MolFromMol2File
from rdkit.Chem.AllChem import SDMolSupplier


class ConfReader:
    def __init__(self):
        self._n_atoms = None
        self._elements = None
        self._coordinates = None
        self._atom_charges = None
        self._n_heavy = None

        self._total_charge = None
        self._total_abs_charge = None

    @property
    def total_charge(self):
        if self._total_charge is None:
            self._total_charge = np.sum(self.atom_charges)
        return self._total_charge

    @property
    def total_abs_charge(self):
        if self._total_abs_charge is None:
            self._total_abs_charge = np.sum(np.abs(self.atom_charges))
        return self._total_abs_charge

    @property
    def atom_charges(self):
        if self._atom_charges is None:
            self._atom_charges = np.asarray([atom.GetFormalCharge() for atom in self.mol.GetAtoms()])
        return self._atom_charges

    @property
    def mol(self):
        raise NotImplemented

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = self.mol.GetNumAtoms()
        return self._n_atoms

    @property
    def elements(self):
        if self._elements is None:
            self._elements = []
            for atom in self.mol.GetAtoms():
                self._elements.append(atom.GetAtomicNum())
        return self._elements

    @property
    def n_heavy(self):
        if self._n_heavy is None:
            n_heavy = 0
            for e in self.elements:
                if e > 1:
                    n_heavy += 1
            self._n_heavy = n_heavy
        return self._n_heavy

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = self.mol.GetConformer().GetPositions()
        return self._coordinates

    def get_basic_dict(self):
        return {"R": torch.as_tensor(self.coordinates).view(-1, 3),
                "Z": torch.as_tensor(self.elements).view(-1),
                "N": torch.as_tensor(self.n_atoms).view(-1)}


class MolReader(ConfReader):
    def __init__(self, mol, addh=False):
        super(MolReader, self).__init__()
        self.addh = addh
        if addh:
            mol = AddHs(mol, addCoords=True)
        self._mol = mol

    @property
    def mol(self):
        return self._mol


class PDBReader(ConfReader):
    """
    Read a PDB file, return coordinates for Pytorch-Geometric dataset preparation
    """
    def __init__(self, pdb_file, dry=False, capped=False, stdaa=False):
        super().__init__()
        # only select protein atoms
        self.dry = dry
        # only select standard amino acids
        self.stdaa = stdaa
        self.pdb_file = pdb_file
        # convert capping ACE and NME to B and J instead of X (unknown)
        self.capped = capped

        self._mol = None
        self._prody_parser = None
        self._res_names = None
        self._atom_names = None
        self._sequence = None
        self._resnums_unique = None
    
    @property
    def atom_charges(self):
        if self._atom_charges is None:
            self._atom_charges = self.prody_parser.getCharges()
        return self._atom_charges

    @property
    def prody_parser(self):
        if self._prody_parser is None:
            self._prody_parser = parsePDB(self.pdb_file)
            if self.dry:
                # You have to convert them to AtomGroups otherwise the residue indices does not work
                self._prody_parser = self._prody_parser.protein.toAtomGroup()
            if self.stdaa:
                self._prody_parser = self._prody_parser.stdaa.toAtomGroup()
        return self._prody_parser

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = self.prody_parser.getCoords()
        return self._coordinates

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = self.prody_parser.numAtoms()
        return self._n_atoms

    @property
    def elements(self):
        if self._elements is None:
            def _atom2num(a):
                symbol = a.getElement()
                if symbol == "":
                    symbol = a.getName()[0]
                    assert symbol in ["C", "H", "O", "N"], a.getName()
                if len(symbol) == 2:
                    # cant beleive I have to do this
                    symbol = symbol[0] + symbol[1].lower()
                atom = ase.Atom(symbol)
                return atom.number
            self._elements = [_atom2num(a) for a in self.prody_parser]
        return self._elements

    @property
    def res_names(self):
        if self._res_names is None:
            self._res_names = self.prody_parser.getResnames()
        return self._res_names

    @property
    def atom_names(self):
        if self._atom_names is None:
            self._atom_names = self.prody_parser.getNames()
        return self._atom_names

    @property
    def resnums_unique(self):
        if self._resnums_unique is None:
            self._resnums_unique = set(self.prody_parser.getResnums())
        return self._resnums_unique

    @property
    def sequence(self):
        if self._sequence is None:
            included = set()
            sequence = ""
            for seq, res_num in zip(self.prody_parser.getSequence(), self.prody_parser.getResnums()):
                if res_num not in included:
                    sequence += seq
                    included.add(res_num)
            res = str(sequence)
            if self.capped:
                assert res[0] == 'X' and res[-1] == 'X'
                res = 'B' + res[1:-1] + 'J'

            self._sequence = res
        return self._sequence


class CGMartiniPDBReader(PDBReader):
    def __init__(self, pdb_file, dry=False):
        super().__init__(pdb_file, dry)
        self._cg_names = None
        self._bead_names = None
        self._vocab = None

    @property
    def vocab(self):
        if self._vocab is None:
            if osp.exists("/scratch/sx801/scripts/Mol3DGenerator/scripts/martini_shaken/MARTINI_VOCAB.json"):
                vocab_path = "/scratch/sx801/scripts/Mol3DGenerator/scripts/martini_shaken/MARTINI_VOCAB.json"
            else:
                vocab_path = "/home/carrot_of_rivia/Documents/PycharmProjects/Mol3DGenerator/scripts/martini_shaken/MARTINI_VOCAB.json"
                assert osp.exists(vocab_path)
            with open(vocab_path) as f:
                self._vocab = json.load(f)
        return self._vocab

    @property
    def bead_names(self):
        if self._bead_names is None:
            self._bead_names = self.atom_names
        return self._bead_names

    @property
    def elements(self):
        if self._elements is None:
            self._elements = [self.vocab[name] for name in self.cg_names]
        return self._elements

    @property
    def cg_names(self):
        if self._cg_names is None:
            self._cg_names = [f"{res}-{name}" for res, name in zip(self.res_names, self.bead_names)]
        return self._cg_names

    def get_aa_bead_info(self):
        batch_idx = self.prody_parser.getResindices()
        info = {}

        prev_res_id = None
        prev_set = set()
        for i, res_id in enumerate(batch_idx):
            if prev_res_id is not None and res_id > prev_res_id:
                tgt_res = self.res_names[i-1]
                if tgt_res not in info.keys():
                    info[tgt_res] = set()
                info[tgt_res].add(frozenset(prev_set))
                prev_set = set()

            prev_res_id = res_id
            prev_set.add(self.bead_names[i])
        
        return info


class PDBLegacyReader(ConfReader):
    """
    Read a PDB file, return coordinates for Pytorch-Geometric dataset preparation
    """
    def __init__(self, pdb_file):
        super().__init__()
        self.pdb_file = pdb_file

        self._mol = None

    @property
    def mol(self):
        if self._mol is None:
            self._mol = MolFromPDBFile(self.pdb_file, removeHs=False)
        return self._mol


class SDFReader(ConfReader):
    def __init__(self, sdf_file, addh=False):
        super().__init__()
        self.sdf_file = sdf_file
        self.addh = addh

        self._mol = None

    @property
    def mol(self):
        if self._mol is None:
            with SDMolSupplier(self.sdf_file, removeHs=False, sanitize=False, strictParsing=False) as supp:
                self._mol = supp[0]
            if self.addh:
                self._mol = AddHs(self._mol, addCoords=True)
        return self._mol


class Mol2Reader(ConfReader):
    def __init__(self, mol2_file):
        super().__init__()
        self.mol2_file = mol2_file

        self._mol = None

    @property
    def mol(self):
        if self._mol is None:
            self._mol = MolFromMol2File(self.mol2_file, removeHs=False, sanitize=False, cleanupSubstructures=False)
        return self._mol


if __name__ == '__main__':
    import json
    tmp = CGMartiniPDBReader("/vast/sx801/PL_dataset/structure_martini/train/binder1_dry/protein/4aba_protein.martini.pdb")
    print(json.dumps(tmp.get_aa_bead_info(), indent=2, default=str))
