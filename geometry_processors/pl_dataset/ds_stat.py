import json
import os
import os.path as osp
from collections import Counter

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

from prody import parsePDB


class ProtDSVisualizer:
    def __init__(self, pdb_files, save_dir, use_tqdm=True) -> None:
        self.save_dir = save_dir
        self.pdb_files = pdb_files
        self.use_tqdm = use_tqdm
        self._file_handles = None

        self._aa_occur = None
        self._atom_occur_prot = None
        self._aa_freq = None
        self._atom_freq_prot = None
        self._num_aas = None
        self._num_atoms_prot = None
        self._num_atoms_wat = None
        self._num_atoms_hetero = None
        self._atom_freq_wat = None
        self._atom_occur_wat = None
        self._atom_occur_hetero = None
        self._atom_freq_hetero = None

        self._info_df = None
        self._counter_summary = None

        self.load_chk()

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.info_df.to_csv(osp.join(self.save_dir, "info_df.csv"), index=False)
        with open(osp.join(self.save_dir, "counter_summary.json"), "w") as f:
            json.dump(self.counter_summary, f, indent=2)
        self.save_dist(self.num_aas, "Number of Amino Acids", osp.join(self.save_dir, "aa_dist.png"))
        self.save_dist(self.num_atoms_prot, "Number of Heavy Atoms in Protein", osp.join(self.save_dir, "prot_atom_dist.png"))
        self.save_dist(self.num_atoms_wat, "Number of Heavy Atoms in Water", osp.join(self.save_dir, "water_atom_dist.png"))
        self.save_dist(self.num_atoms_hetero, "Number of Heavy Atoms in Hetero", osp.join(self.save_dir, "hetero_atom_dist.png"))
        plt.figure(figsize=(10, 5))
        self.bar_plot(self.aa_occur, "Amimo Acid", "Occurence", osp.join(self.save_dir, "aa_occur.png"))
        plt.figure(figsize=(10, 5))
        self.bar_plot(self.aa_freq, "Amimo Acid", "Frequency", osp.join(self.save_dir, "aa_freq.png"))
        self.bar_plot(self.atom_occur_prot, "Atom", "Occurence", osp.join(self.save_dir, "atom_occur_prot.png"))
        self.bar_plot(self.atom_freq_prot, "Atom", "Frequency", osp.join(self.save_dir, "atom_freq_prot.png"))

    def load_chk(self):
        chk_file = osp.join(self.save_dir, "counter_summary.json")
        if not osp.exists(chk_file):
            return
        
        print(f"loading checkpoint from {chk_file}")
        with open(chk_file) as f:
            chk = json.load(f)
        for key in chk:
            setattr(self, "_"+key, chk[key])

        chk_df = pd.read_csv(osp.join(self.save_dir, "info_df.csv"))
        self._num_aas = chk_df["num_aas"].values
        self._num_atoms_prot = chk_df["protein_atoms"].values
        self._num_atoms_wat = chk_df["water_atoms"].values
        self._num_atoms_hetero = chk_df["hetero_atoms"].values
        self._file_handles = chk_df["file_handle"].values
    
    @staticmethod
    def bar_plot(counter, x, y, out):
        data = {x: [key for key in counter], y: [counter[key] for key in counter]}
        data = pd.DataFrame(data).sort_values(by=x, axis=0)
        sns.barplot(data=data, x=x, y=y, color="salmon", saturation=.5, log=True, edgecolor=".2")
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
    
    @staticmethod
    def save_dist(data_list, x_label, out):
        info = {x_label: data_list}
        info = pd.DataFrame(info)
        if max(data_list) < 100:
            bins = list(range(max(data_list) + 2))
        else:
            bins = 50
        sns.histplot(data=info, bins=bins, x=x_label)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        
    @property
    def info_df(self):
        if self._info_df is None:
            out_info = {"file_handle": self.file_handles,
                        "num_aas": self.num_aas,
                        "protein_atoms": self.num_atoms_prot,
                        "water_atoms": self.num_atoms_wat,
                        "hetero_atoms": self.num_atoms_hetero}
            info_df = pd.DataFrame(out_info)
            self._info_df = info_df
        return self._info_df

    @property
    def counter_summary(self):
        if self._counter_summary is None:
            summary = {}
            for key in ["aa_occur", "aa_freq", "atom_occur_prot", "atom_freq_prot", "atom_occur_wat",
                    "atom_freq_wat", "atom_occur_hetero", "atom_freq_hetero"]:
                summary[key] = dict(getattr(self, key))
            self._counter_summary = summary
        return self._counter_summary

    @property
    def aa_occur(self):
        if self._aa_occur is None:
            logger = logging.getLogger(".prody")
            logger.setLevel(logging.CRITICAL)
            
            aa_occur = Counter()
            aa_freq = Counter()
            atom_occur_prot = Counter()
            atom_freq_prot = Counter()
            atom_occur_wat = Counter()
            atom_freq_wat = Counter()
            atom_occur_hetero = Counter()
            atom_freq_hetero = Counter()
            num_aas = []
            num_atoms_prot = []
            num_atoms_wat = []
            num_atoms_hetero = []

            if self.use_tqdm:
                pdb_files = tqdm(self.pdb_files)
            else:
                pdb_files = self.pdb_files
            for pdb in pdb_files:
                structure = parsePDB(pdb)
                protein = structure.select("protein")
                water = structure.select("water")
                hetero = structure.select("hetero")

                aas_all = protein.getResnames()
                aas_idx = protein.getResindices()
                prev = -1
                aas = []
                for aa, idx in zip(aas_all, aas_idx):
                    if idx != prev:
                        aas.append(aa)
                    prev = idx
                aa_freq.update(aas)
                aa_occur.update(set(aas))
                num_aas.append(len(aas))
                atoms_prot = protein.getElements()
                atom_freq_prot.update(atoms_prot)
                atom_occur_prot.update(set(atoms_prot))
                num_atoms_prot.append(len(atoms_prot))

                if water is not None:
                    atoms_wat = water.getElements()
                    atom_freq_wat.update(atoms_wat)
                    atom_occur_wat.update(set(atoms_wat))
                    num_atoms_wat.append(len(atoms_wat))
                else:
                    num_atoms_wat.append(0)

                if hetero is not None:
                    atoms_hetero = hetero.getElements()
                    atom_freq_hetero.update(atoms_hetero)
                    atom_occur_hetero.update(set(atoms_hetero))
                    num_atoms_hetero.append(len(atoms_hetero))
                else:
                    num_atoms_hetero.append(0)

            self._aa_occur = aa_occur
            self._atom_occur_prot = atom_occur_prot
            self._aa_freq = aa_freq
            self._atom_freq_prot = atom_freq_prot
            self._num_aas = num_aas
            self._num_atoms_prot = num_atoms_prot
            self._num_atoms_wat = num_atoms_wat
            self._num_atoms_hetero = num_atoms_hetero

            self._atom_freq_wat = atom_freq_wat
            self._atom_occur_wat = atom_occur_wat
            self._atom_freq_hetero = atom_freq_hetero
            self._atom_occur_hetero = atom_occur_hetero
        return self._aa_occur

    @property
    def atom_occur_prot(self):
        if self._atom_occur_prot is None:
            __ = self.aa_occur
        return self._atom_occur_prot

    @property
    def atom_freq_prot(self):
        if self._atom_freq_prot is None:
            __ = self.aa_occur
        return self._atom_freq_prot

    @property
    def atom_occur_hetero(self):
        if self._atom_occur_hetero is None:
            __ = self.aa_occur
        return self._atom_occur_hetero

    @property
    def atom_freq_hetero(self):
        if self._atom_freq_hetero is None:
            __ = self.aa_occur
        return self._atom_freq_hetero

    @property
    def atom_occur_wat(self):
        if self._atom_occur_wat is None:
            __ = self.aa_occur
        return self._atom_occur_wat

    @property
    def atom_freq_wat(self):
        if self._atom_freq_wat is None:
            __ = self.aa_occur
        return self._atom_freq_wat

    @property
    def aa_freq(self):
        if self._aa_freq is None:
            __ = self.aa_occur
        return self._aa_freq

    @property
    def num_aas(self):
        if self._num_aas is None:
            __ = self.aa_occur
        return self._num_aas

    @property
    def num_atoms_prot(self):
        if self._num_atoms_prot is None:
            __ = self.aa_occur
        return self._num_atoms_prot

    @property
    def num_atoms_wat(self):
        if self._num_atoms_wat is None:
            __ = self.aa_occur
        return self._num_atoms_wat

    @property
    def num_atoms_hetero(self):
        if self._num_atoms_hetero is None:
            __ = self.aa_occur
        return self._num_atoms_hetero

    @property
    def file_handles(self):
        if self._file_handles is None:
            res = [osp.basename(f).split(".pdb")[0] for f in self.pdb_files]
            self._file_handles = res
        return self._file_handles
