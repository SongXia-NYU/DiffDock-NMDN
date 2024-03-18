import json
import os
import os.path as osp
import subprocess
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import random

from prody import parsePDB, writePDB, AtomGroup



class ProteinSplitter:
    def __init__(self, src_pdb, root, frag_size=1, init_skip: int = "auto", inter_skip=0, debug=False, seed=None,
                 use_rd=True, tgt_n_aa=None, tgt_n_frags=None, n_sample=None):
        self.n_sample = n_sample
        if tgt_n_aa is not None:
            assert frag_size == 1, frag_size
        self.tgt_n_aa = tgt_n_aa
        self.tgt_n_frags = tgt_n_frags
        if tgt_n_frags is not None:
            assert init_skip == "auto", "init_skip and inter_skip cannot be assigned when tgt_n_frags is not None"
            assert inter_skip == 0, "init_skip and inter_skip cannot be assigned when tgt_n_frags is not None"
        # I refuse to explain this because it is magic
        self.magic_factor = 2

        self.use_rd = use_rd
        random.seed(seed)
        self.inter_skip = inter_skip
        self.init_skip = init_skip
        if init_skip == "auto":
            self.init_skip = frag_size - 1
        self.frag_size = frag_size
        self.debug = debug
        self.src_pdb = src_pdb
        self._root = root

        self._save_root = None
        self._atoms = None
        self._file_handle = None
        self._resnums = None
        self._stored_selection = {}

        # only used when tgt_n_aa is not None to sample specific number of AAs
        self._aa_freq = None
        self._aa2seq = None

        self._is_active = False

    def run(self, locators=None):
        if locators is None:
            locators = [np.random.choice(self.resnums[1:-self.frag_size], 1).item() for __ in range(self.n_sample)]
        for loc in locators:
            end = loc + self.frag_size
            if loc - 1 < self.resnums[0] or end > self.resnums[-1]:
                continue

            ace_co = self.select(f"name C O resnum {loc - 1}")
            ace_ca = self.select(f"name CA resnum {loc - 1}")
            ace_ca.setNames("CH3")
            ace = (ace_co + ace_ca).toAtomGroup()
            ace.setResnames("ACE")

            main_sel = f"resnum {loc}:{end}" if self.frag_size > 1 else f"resnum {loc}"
            main_sel = self.select(main_sel).toAtomGroup()

            nme_n = self.select(f"name N resnum {end}")
            nme_ca = self.select(f"name CA resnum {end}")
            nme_ca.setNames("C")
            nme = (nme_n + nme_ca).toAtomGroup()
            nme.setResnames("NME")

            all_sel = ace + main_sel + nme
            all_sel.setTitle(main_sel)

            self.write_pdb(loc, all_sel)

    def adjust_skip(self):
        if self.tgt_n_frags is not None:
            n_aa_ds = sum([self.aa_freq[aa] for aa in self.aa_freq.keys()])
            n_aa_ds = n_aa_ds // self.magic_factor
            expected_skip = n_aa_ds // self.tgt_n_frags

            extra_skip = expected_skip - self.frag_size
            if self.use_rd:
                extra_skip = 2 * extra_skip - 1
            self.init_skip = extra_skip
            self.inter_skip = extra_skip

    @property
    def resnums(self):
        if self._resnums is None:
            all_resnums = set(self.atoms.getResnums())
            all_resnums = [int(i) for i in all_resnums]
            all_resnums.sort()
            self._resnums = np.asarray(all_resnums, dtype=int)
        return self._resnums

    def run_old(self):
        """
        An older implementation at atom-level. Harder to read and less-intuitive. So I rewrote it
        :return:
        """
        self.adjust_skip()

        if self.debug:
            if len(glob(osp.join(self.save_root, "*"))) > 0:
                subprocess.run(f"rm {osp.join(self.save_root, '*')} ", shell=True, check=True)
            writePDB(osp.join(self.save_root, f"{self.file_handle}.pdb"), self.atoms)

        # the residue id of previous AA
        prev_res = 0
        skip_counter = 0
        next_skip = self.get_n_init_skip()
        # the selected atoms: AtomGroup
        sel_atoms = None
        # number of selected residues
        n_sel_res = 0

        for atom in self.atoms:
            # the residue id of the current atom
            curr_res = atom.getResindex()
            # does the current atom belong to a new residue or the previous residue
            is_new_res = (curr_res == prev_res + 1)

            if is_new_res:
                # if the atom belongs to a new residue, either update selected atoms or skip counters
                # depending on the activation state
                prev_res = curr_res
                if self.is_active:
                    n_sel_res += 1
                else:
                    skip_counter += 1
            else:
                assert curr_res == prev_res, f"{curr_res}, {prev_res}"

            if not self.is_active:
                # if not active, check skip counters to either activate or skip this atom
                if skip_counter == next_skip:
                    self.activate()
                    skip_counter = 0
                else:
                    sel_atoms = None
                    continue

            if not is_new_res or n_sel_res < self.frag_size:
                # still in the same residue
                if sel_atoms is None:
                    sel_atoms = atom
                else:
                    sel_atoms = sel_atoms + atom
                    sel_atoms.setTitle(f"title :-) ")
                continue

            self.write_pdb(curr_res, sel_atoms)
            # re-initialize
            sel_atoms = atom
            n_sel_res = 0
            next_skip = self.get_n_inter_skip()
            self.deactivate()

            if self.debug and curr_res >= 5:
                # early stopping in debug mode
                break

    def select(self, selection):
        if selection in self._stored_selection:
            return self._stored_selection[selection]

        selected = self.atoms.select(f"{selection}")
        self._stored_selection[selection] = selected
        return selected

    def write_pdb(self, current_res, atoms):
        if atoms is None:
            return

        if self.tgt_n_aa:
            rd = random.randint(1, self.aa_freq[atoms[0].getSequence()])
            if rd > self.tgt_n_aa * self.magic_factor:
                return

        out_p = osp.join(self.save_root, f"{self.file_handle}@{self.frag_size}.{current_res}.pdb")
        writePDB(out_p, atoms)
        # out_tmp_p = osp.join(self.save_root, f"{self.file_handle}@{self.frag_size}.{curr_res}.tmp.pdb")
        # fix_cmd = f"pdbfixer '{out_tmp_p}' --add-atoms=heavy --replace-nonstandard --output '{out_p}'"
        # subprocess.run(fix_cmd, shell=True, check=True)
        # if not self.debug:
        #     subprocess.run(f"rm '{out_tmp_p}'", shell=True, check=True)

    def get_n_init_skip(self):
        if self.init_skip == 0:
            return 0
        elif not self.use_rd:
            return self.init_skip
        return random.randint(0, self.init_skip)

    def get_n_inter_skip(self):
        if self.inter_skip == 0:
            return 0
        elif not self.use_rd:
            return self.inter_skip
        return random.randint(0, self.inter_skip)

    @property
    def aa_freq(self):
        if self._aa_freq is None:
            root = "/home/carrot_of_rivia/Documents/disk/datasets/AF-SwissProt-stats/"
            if osp.exists("/scratch/sx801"):
                root = "/scratch/sx801/scripts/Mol3DGenerator/scripts/AF-SwissProt/AF-SwissProt-stats/"
            with open(f"{root}counter_summary.json") as f:
                info_raw = json.load(f)
            out = {}
            for aa in info_raw["aa_freq"]:
                out[self.aa2seq[aa]] = info_raw["aa_freq"][aa]
            self._aa_freq = out
        return self._aa_freq

    @property
    def aa2seq(self):
        if self._aa2seq is None:
            self._aa2seq = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                            'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        return self._aa2seq

    @property
    def is_active(self):
        return self._is_active

    def activate(self):
        assert not self.is_active
        self._is_active = True

    def deactivate(self):
        assert self.is_active
        self._is_active = False

    @property
    def file_handle(self):
        if self._file_handle is None:
            self._file_handle = osp.basename(self.src_pdb).split(".pdb")[0]
        return self._file_handle

    @property
    def atoms(self):
        if self._atoms is None:
            atoms = parsePDB(self.src_pdb)
            self._atoms = atoms
        return self._atoms

    @property
    def save_root(self):
        if self._save_root is None:
            save_root = osp.join(self._root, self.file_handle)
            os.makedirs(save_root, exist_ok=True)
            self._save_root = save_root
        return self._save_root


if __name__ == '__main__':
    pass
