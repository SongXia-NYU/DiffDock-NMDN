from typing import List, Tuple, Optional
from geometry_processors.pl_dataset.ConfReader import ConfReader, PDBQTReader, PDBStreamReader, SDFReader, Mol2Reader, PDBReader, MolReader, PDBLegacyReader, CGMartiniPDBReader
from geometry_processors.pl_dataset.csv2input_list import MPInfo

import numpy as np


class ConfReaderFactory:
    def __init__(self, info: MPInfo, protein_reader_args: Optional[dict] = None, ligand_reader_args: Optional[dict] = None, **kwargs) -> None:
        self.info: MPInfo = info

        self.protein_reader_args: Optional[dict] = protein_reader_args
        self.ligand_reader_args: Optional[dict] = ligand_reader_args

        self.martini: bool = kwargs.pop("martini", False)
        self.legacy: bool = kwargs.pop("legacy", False)
        self.neutral: bool = kwargs.pop("neutral", False)

        assert not kwargs, kwargs

        self._chains_chain1_reader = None
        self._chains_chain2_reader = None
        self._dockground2_chain1_reader = None
        self._dockground2_chain2_reader = None

    def get_lig_reader(self) -> Tuple[ConfReader, str]:
        reader_args = self.ligand_reader_args if self.ligand_reader_args is not None else {}
        for key in ["overwrite_lig_pos", "remove_h"]:
            prop = getattr(self.info, key)
            reader_args[key] = prop
        if self.info.ligand_sdf is not None:
            ligand_reader = SDFReader(self.info.ligand_sdf, **reader_args)
            ligand_file = self.info.ligand_sdf
        elif self.info.ligand_mol2 is not None:
            ligand_reader = Mol2Reader(self.info.ligand_mol2, **reader_args)
            ligand_file = self.info.ligand_mol2
        elif self.info.ligand_pdb is not None:
            ligand_reader = PDBReader(self.info.ligand_pdb, **reader_args)
            ligand_file = self.info.ligand_pdb
        elif self.info.ligand_pdbqt is not None:
            ligand_reader = PDBQTReader(self.info.ligand_pdbqt, **reader_args)
            ligand_file = self.info.ligand_pdbqt
        else:
            assert self.info.mol is not None
            ligand_reader = MolReader(self.info.mol, **reader_args)
            ligand_file = self.info.ligand_id
        if self.neutral:
            if np.abs(ligand_reader.atom_charges).sum() != 0:
                raise ValueError("Atom charge is not None!")
            
        if self.info.overwrite_lig_pos is not None:
            assert isinstance(ligand_reader, (SDFReader, Mol2Reader))
        return ligand_reader, ligand_file
    
    def get_prot_reader(self) -> Tuple[ConfReader, str]:
        return self._get_prot_reader_by_key("protein_pdb")
    
    def get_chain1_reader(self) -> Tuple[ConfReader, str]:
        if self.info.chains_combined_pdb is not None:
            self.parse_chains_combined_pdb()
            return self._chains_chain1_reader, self.info.chains_combined_pdb + ".chain1"
        # manually parse dockground2 combined pdb file
        if self.info.dockground2_combined_pdb is not None:
            self.parse_dockground2_combined_pdb()
            return self._dockground2_chain1_reader
        return self._get_prot_reader_by_key("protein_chain1_pdb")
    
    def get_chain2_reader(self) -> Tuple[ConfReader, str]:
        if self.info.chains_combined_pdb is not None:
            self.parse_chains_combined_pdb()
            return self._chains_chain2_reader, self.info.chains_combined_pdb + ".chain2"
        # manually parse dockground2 combined pdb file
        if self.info.dockground2_combined_pdb is not None:
            self.parse_dockground2_combined_pdb()
            return self._dockground2_chain2_reader
        return self._get_prot_reader_by_key("protein_chain2_pdb")
    
    def parse_chains_combined_pdb(self):
        if self._chains_chain2_reader is not None: return

        reader_args = self.protein_reader_args if self.protein_reader_args is not None else {}
        protein_reader = PDBReader(self.info.chains_combined_pdb, **reader_args)
        chains = [chain for chain in protein_reader.prody_parser.iterChains()]
        assert len(chains) == 2, len(chains)
        # chain 1
        protein_reader.set_prody_parser(chains[0].toAtomGroup())
        self._chains_chain1_reader = protein_reader
        # chain 2
        protein_reader = PDBReader(self.info.chains_combined_pdb, **reader_args)
        protein_reader.set_prody_parser(chains[1].toAtomGroup())
        self._chains_chain2_reader = protein_reader
    
    def parse_dockground2_combined_pdb(self):
        if self._dockground2_chain2_reader is not None: return

        chain1_stream, chain2_stream = self.dockground_combined_pdb2streams(self.info.dockground2_combined_pdb)
        reader_args = self.protein_reader_args if self.protein_reader_args is not None else {}
        chain1_fl = self.info.dockground2_combined_pdb+".chain1"
        self._dockground2_chain1_reader = PDBStreamReader(chain1_stream, chain1_fl, **reader_args), chain1_fl
        chain2_fl = self.info.dockground2_combined_pdb+".chain2"
        self._dockground2_chain2_reader = PDBStreamReader(chain2_stream, chain2_fl, **reader_args), chain2_fl
    
    @staticmethod
    def dockground_combined_pdb2streams(pdb_file: str):
        with open(pdb_file) as f:
            raw_lines: List[str] = f.readlines()
        chain1_lines: List[str] = []
        chain2_lines: List[str] = []
        is_chain2: bool = False
        for line in raw_lines:
            if is_chain2: chain2_lines.append(line)
            else: chain1_lines.append(line)
            if line.startswith("END"): is_chain2 = True
        return FakeFileStream(chain1_lines), FakeFileStream(chain2_lines)
    
    def _get_prot_reader_by_key(self, key: str) -> Tuple[ConfReader, str]:
        protein_file: str = getattr(self.info, key)

        reader_args = self.protein_reader_args if self.protein_reader_args is not None else {}
        if protein_file.endswith(".martini.pdb"):
            assert self.martini
        if self.legacy:
            protein_reader = PDBLegacyReader(protein_file)
        elif self.martini:
            protein_reader = CGMartiniPDBReader(protein_file, **reader_args)
        else:
            protein_reader = PDBReader(protein_file, **reader_args)

        if self.neutral:
            if np.abs(protein_reader.atom_charges).sum() != 0:
                raise ValueError("Atom charge is not None!")
        return protein_reader, protein_file


class FakeFileStream:
    def __init__(self, lines: List[str]) -> None:
        self.lines = lines
    
    def readlines(self):
        return self.lines
