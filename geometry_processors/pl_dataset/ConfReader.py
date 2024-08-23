import os.path as osp

import json
from typing import Dict, List, Optional, Union
import ase
import numpy as np
from prody import AtomGroup
import torch
from prody.proteins import parsePDB, parsePDBStream
from rdkit.Chem import MolFromPDBFile, AddHs, MolFromMol2File
from rdkit.Chem.AllChem import SDMolSupplier

from geometry_processors.lazy_property import lazy_property


class ConfReader:
    def __init__(self, overwrite_lig_pos: Union[np.ndarray, torch.Tensor]=None, remove_h: bool=False):
        self._n_atoms = None
        self._elements = None
        self._coordinates = None
        self._atom_charges = None
        self._n_heavy = None

        self._total_charge = None
        self._total_abs_charge = None

        self.overwrite_lig_pos: Union[np.ndarray, torch.Tensor] = overwrite_lig_pos
        self.remove_h = remove_h

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
    def coordinates(self) -> np.ndarray:
        if self._coordinates is None:
            self._coordinates = self.mol.GetConformer().GetPositions()
        return self._coordinates

    def get_basic_dict(self):
        out_dict = {
            "R": torch.as_tensor(self.coordinates).view(-1, 3),
            "Z": torch.as_tensor(self.elements).view(-1),
            "N": torch.as_tensor(self.n_atoms).view(-1)}
        if self.remove_h:
            masker: torch.BoolTensor = (out_dict["Z"] != 1)
            out_dict["R"] = out_dict["R"][masker, :]
            out_dict["Z"] = out_dict["Z"][masker]
            out_dict["N"] = masker.sum()
        if self.overwrite_lig_pos is not None:
            out_dict["R"] = torch.as_tensor(self.overwrite_lig_pos).view(-1, 3)
        return out_dict


class MolReader(ConfReader):
    def __init__(self, mol, addh=False, **kwargs):
        super(MolReader, self).__init__(**kwargs)
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
    def __init__(self, pdb_file, dry=False, capped=False, stdaa=False, force_polarh=False, **kwargs):
        super().__init__(**kwargs)
        # only select protein atoms
        self.dry = dry
        # only select standard amino acids
        self.stdaa = stdaa
        self.pdb_file = pdb_file
        # convert capping ACE and NME to B and J instead of X (unknown)
        self.capped = capped
        # force removing non-polar hydrogens. Input structures must have hydrogen added.
        self.force_polarh: bool = force_polarh

        self._mol = None
        self._prody_parser = None
        self._res_names = None
        self._atom_names = None
        self._sequence = None
        self._resnums_unique = None
        self._vocab = None

    def get_padding_style_dict(self):
        """
        Process protein in a padding style: the protein_R is [N_aa, N_max_atom_per_aa, 3] 
        and protein_Z is [N_aa, N_max_atom_per_aa]
        """
        R = []
        Z = []
        N = []
        res_nums: List[int] = []
        # get the un-padded version first
        for res_num, res in enumerate(self.prody_parser.iterResidues()):
            this_r = res.getCoords()
            this_z = np.asarray([self._atom2num(a) for a in res])
            this_n = len(res)

            R.append(this_r)
            Z.append(this_z)
            N.append(this_n)
            res_nums.append(res_num)
        N = np.asarray(N)
        res_nums = np.asarray(res_nums)
        n_max = np.max(N)

        # pad the coordinates and atomic numbers
        R_padded = []
        for r in R:
            if r.shape[0] < n_max:
                r = np.pad(r, ((0, n_max - r.shape[0]), (0, 0)),
                     mode="constant", constant_values=np.nan)
            R_padded.append(r.reshape(1, -1, 3))
        
        Z_padded = []
        for z in Z:
            if z.shape[0] < n_max:
                z = np.pad(z, (0, n_max - z.shape[0]), mode="constant", constant_values=-1)
            Z_padded.append(z.reshape(1, -1))
        R_padded = np.concatenate(R_padded)
        Z_padded = np.concatenate(Z_padded)
        res_dict = {"R": R_padded, "Z": Z_padded, "N": N, "res_num": res_nums}
        return res_dict
    
    @lazy_property
    def padding_style_dict(self) -> Dict[str, np.ndarray]:
        return self.get_padding_style_dict()
    
    @property
    def atom_charges(self):
        if self._atom_charges is None:
            self._atom_charges = self.prody_parser.getCharges()
        return self._atom_charges

    @property
    def prody_parser(self) -> AtomGroup:
        if self._prody_parser is None:
            self._prody_parser = parsePDB(self.pdb_file)
            if self.dry:
                # You have to convert them to AtomGroups otherwise the residue indices does not work
                self._prody_parser = self._prody_parser.protein.toAtomGroup()
            if self.stdaa:
                self._prody_parser = self._prody_parser.stdaa.toAtomGroup()
            if self.force_polarh:
                self._prody_parser = self.remove_nonpolar(self._prody_parser)
        return self._prody_parser

    @staticmethod
    def remove_nonpolar(ag: AtomGroup) -> AtomGroup:
        ag.inferBonds()
        ag = ag.select("not (hydrogen and bonded 1 to carbon)")
        return ag.toAtomGroup()
    
    def set_prody_parser(self, parser: AtomGroup) -> None:
        self._prody_parser = parser

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
            self._elements = [self._atom2num(a) for a in self.prody_parser]
        return self._elements

    @staticmethod
    def _atom2num(a):
        symbol = a.getElement()
        if symbol == "":
            symbol = a.getName()[0]
            assert symbol in ["C", "H", "O", "N", "S", "P"], a.getName()
        if len(symbol) == 2:
            # cant beleive I have to do this
            symbol = symbol[0] + symbol[1].lower()
        atom = ase.Atom(symbol)
        return atom.number

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
    def vocab(self):
        if self._vocab is None:
            vocab_path = osp.join(osp.dirname(__file__), "MARTINI_VOCAB.json")
            assert osp.exists(vocab_path)
            with open(vocab_path) as f:
                self._vocab = json.load(f)
        return self._vocab

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


class PDBHeteroReader(PDBReader):
    """
    Read a PDB file and return a heterogenous graph containing information about protein, water and ions.
    """
    def __init__(self, pdb_file, **kwargs):
        super().__init__(pdb_file, dry=True, capped=False, stdaa=False, **kwargs)

    def parse_hetero_graph(self) -> Dict[str, Dict[str, torch.Tensor]]:
        info = {"protein": self.padding_style_dict}
        info["ion"] = self.parse_ion_info()
        info["water"] = self.parse_water_info()
        return info
    
    def parse_ion_info(self) -> Dict[str, torch.Tensor]:
        if self.ion_ag is None:
            return {"R": torch.zeros([0, 3]).float(), "Z": torch.zeros([0]).long()}
        
        coords = []
        atom_nums = []
        for res_num, res in enumerate(self.ion_ag.iterResidues()):
            coords.append(res.getCoords())
            atom_nums.append([self._atom2num(a) for a in res])
        return {"R": torch.as_tensor(np.concatenate(coords, axis=0)), 
                "Z": torch.as_tensor(atom_nums)}
    
    def parse_water_info(self) -> Dict[str, torch.Tensor]:
        if self.water_ag is None:
            return {"R": torch.zeros([0, 3, 3]).float(), "Z": torch.zeros([0, 3]).long()}
        
        coords = []
        atom_nums = []
        for res_num, res in enumerate(self.water_ag.iterResidues()):
            # [3, 3]
            water_coords: np.ndarray = res.getCoords()
            if water_coords.shape[0] != 3:
                # print("Imcomplete water molecule!! discarding...")
                continue
            coords.append(water_coords.reshape(1, 3, 3))
            atom_nums.append([self._atom2num(a) for a in res])
        if len(coords) == 0:
            # if all water molecules are incomplete, return empty matrix
            return {"R": torch.zeros([0, 3, 3]).float(), "Z": torch.zeros([0, 3]).long()}
        return {"R": torch.as_tensor(np.concatenate(coords, axis=0)), 
                "Z": torch.as_tensor(atom_nums)}

    @lazy_property
    def protein_ag(self) -> AtomGroup:
        return self.prody_parser

    @lazy_property
    def water_ag(self) -> Optional[AtomGroup]:
        sel = parsePDB(self.pdb_file).water
        if sel is None:
            return None
        return sel.toAtomGroup()

    @lazy_property
    def ion_ag(self) -> Optional[AtomGroup]:
        sel = parsePDB(self.pdb_file).ion
        if sel is None:
            return None
        return sel.toAtomGroup()
    

class PDBStreamReader(PDBReader):
    def __init__(self, stream, file_handle, dry=False, capped=False, stdaa=False, **kwargs):
        self.stream = stream
        super().__init__(file_handle, dry, capped, stdaa, **kwargs)

    @lazy_property
    def prody_parser(self):
        prody_parser = parsePDBStream(self.stream)
        if self.dry:
            # You have to convert them to AtomGroups otherwise the residue indices does not work
            prody_parser = prody_parser.protein.toAtomGroup()
        if self.stdaa:
            prody_parser = prody_parser.stdaa.toAtomGroup()
        return prody_parser


class CGMartiniPDBReader(PDBReader):
    def __init__(self, pdb_file, dry=False):
        super().__init__(pdb_file, dry)
        self._cg_names = None
        self._bead_names = None

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


class BetaAtomPDBReader(PDBReader):
    def __init__(self, pdb_file, dry=False, capped=False, stdaa=False):
        super().__init__(pdb_file, dry, capped, stdaa)

    @property
    def elements(self):
        n = self.n_atoms
        res = [self.vocab["beta_atom"]] * n
        return res


class PDBLegacyReader(ConfReader):
    """
    Read a PDB file, return coordinates for Pytorch-Geometric dataset preparation
    """
    def __init__(self, pdb_file, **kwargs):
        super().__init__(**kwargs)
        self.pdb_file = pdb_file

        self._mol = None

    @property
    def mol(self):
        if self._mol is None:
            self._mol = MolFromPDBFile(self.pdb_file, removeHs=False)
        return self._mol


class SDFReader(ConfReader):
    def __init__(self, sdf_file, addh=False, **kwargs):
        super().__init__(**kwargs)
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
    def __init__(self, mol2_file, **kwargs):
        super().__init__(**kwargs)
        self.mol2_file = mol2_file

        self._mol = None

    @property
    def mol(self):
        if self._mol is None:
            self._mol = MolFromMol2File(self.mol2_file, removeHs=False, sanitize=False, cleanupSubstructures=False)
        return self._mol
    
class PDBQTReader(ConfReader):
    def __init__(self, pdbqt_file, **kwargs):
        super().__init__(**kwargs)
        self.pdbqt_file = pdbqt_file
        self._mol = None

    @property
    def mol(self):
        from meeko import PDBQTMolecule
        from meeko import RDKitMolCreate
        if self._mol is None:
            pdbqt_mol = PDBQTMolecule.from_file(self.pdbqt_file, skip_typing=True)
            rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
            self._mol = rdkitmol_list[0]
        return self._mol


if __name__ == '__main__':
    fname = "/scratch/xd638/20240407_DeepAccNet_unzip/test/new_pdbs/1af7A_0/1af7A_0_100.pdb"
    tmp = PDBReader(fname, dry=True, force_polarh=True)
    from prody import writePDB
    writePDB("/scratch/sx801/temp/1af7A_0_100.polarh.pdb", tmp.prody_parser)
    print(tmp.get_padding_style_dict())
