from glob import glob
import pandas as pd
import os
import os.path as osp
import tqdm
import copy
import logging
from tqdm.contrib.concurrent import process_map
import seaborn as sns
import matplotlib.pyplot as plt

from geometry_processors.pl_dataset.ConfReader import PDBReader


def runner(args):
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    checker = MartiniChecker(*args)
    checker.run()


def diff_reader(csv):
    df = pd.read_csv(csv)
    og_seq = df["og_sequence"].item()
    martini_seq = df["martini_sequence"].item()
    return len(og_seq) - len(martini_seq)


class BatchMartiniChecker:
    def __init__(self, pdbs, martini_pdbs, save_dir):
        self.pdbs = pdbs
        self.martini_pdbs = martini_pdbs
        self.save_dir = save_dir

    def run(self):
        single_folder = osp.join(self.save_dir, "single_csvs")
        if not osp.exists(single_folder):
            os.makedirs(single_folder, exist_ok=False)
            checker_parms = [(pdb, martini_pdb, single_folder) for pdb, martini_pdb in zip(self.pdbs, self.martini_pdbs)]
            process_map(runner, checker_parms, chunksize=10)

        diffs = process_map(diff_reader, glob(osp.join(single_folder, "*.csv")))
        ax = sns.histplot(diffs)
        ax.set(xlabel="len_seq(Original)-len_seq(Martini)")
        plt.tight_layout()
        plt.savefig(osp.join(self.save_dir, "diff_hist.png"))


class MartiniChecker:
    def __init__(self, pdb, martini_pdb, save_dir) -> None:
        self.pdb = pdb
        self.martini_pdb = martini_pdb
        self.save_dir = save_dir

        self.og_reader = PDBReader(pdb, dry=True)
        self.martini_reader = PDBReader(martini_pdb)

    def run(self):
        if len(self.og_reader.sequence) != len(self.martini_reader.sequence.strip("X")):
            res = {"og_pdb": osp.basename(self.pdb), "martini_pdb": osp.basename(self.martini_pdb)}
            res["og_missing"] = str(self.martini_reader.resnums_unique.difference(self.og_reader.resnums_unique))
            martini_missing = self.og_reader.resnums_unique.difference(self.martini_reader.resnums_unique)
            martini_missing_seq = ""
            for i in martini_missing:
                martini_missing_seq += self.og_reader.sequence[i]
            res["martini_missing_seq"] = martini_missing_seq
            res["martini_missing"] = str(martini_missing)
            res["og_sequence"] = self.og_reader.sequence
            res["martini_sequence"] = self.martini_reader.sequence
            res = pd.DataFrame(res, index=[0])
            
            file_handle = osp.basename(self.pdb).split("amber.H")[0]
            res.to_csv(osp.join(self.save_dir, f"{file_handle}.csv"), index=False)

