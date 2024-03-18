import logging
import os.path as osp
import pdb
import pandas as pd
from glob import glob
from tqdm.contrib.concurrent import process_map

from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.all2single_pygs import ProteinProcessor
from geometry_processors.misc import solv_num_workers


class DSPreparator:
    def __init__(self, protein_folder, csv_templ, p_min, p_max, pyg_root, use_martini=False):
        self.protein_folder = protein_folder
        self.csv_templ = csv_templ
        self.p_min = p_min
        self.p_max = p_max
        self.pyg_root = pyg_root
        self.use_martini = use_martini

    def run(self, debug=False):
        pdb_files = glob(osp.join(self.protein_folder, "*.pdb"))
        if debug:
            pdb_files = pdb_files[:5]
        n_cpu_avail, n_cpu, num_workers = solv_num_workers()
        mp_args = [(f, self.csv_templ, self.p_min, self.p_max, self.pyg_root, self.use_martini) for f in pdb_files]
        errors = process_map(mp_process_prot, mp_args, chunksize=20, max_workers=num_workers)
        errors = [e for e in errors if e is not None]
        return errors

def fl_extractor(f):
    return osp.basename(f).split(".amber.H.pdb")[0]

def mp_process_prot(args):
    pdb_f, csv_templ, p_min, p_max, pyg_root, use_martini = args
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)
    prop_dict = {}
    fl = fl_extractor(pdb_f)

    try:
        energy = {}
        for phase in ["gas", "water", "octanol"]:
            this_e = pd.read_csv(csv_templ.format(phase=phase, fl=fl))["ENERGY"].item()
            energy[phase] = float(this_e)
            if energy[phase] < p_min or energy[phase] > p_max:
                raise ValueError(f"Property not within range [{p_min}, {p_max}]: {energy[phase]}")
        prop_dict["gasEnergy"] = energy["gas"]
        prop_dict["watEnergy"] = energy["water"]
        prop_dict["octEnergy"] = energy["octanol"]
        prop_dict["CalcSol"] = energy["water"] - energy["gas"]
        prop_dict["CalcOct"] = energy["octanol"] - energy["gas"]
        prop_dict["watOct"] = energy["water"] - energy["octanol"]

        mp_info = MPInfo(protein_pdb=pdb_f, cutoff_protein=10., pyg_name=f"{pyg_root}/{fl}.pth", **prop_dict)

        processor = ProteinProcessor(mp_info, martini=use_martini)

        processor.process_single_entry()
    except Exception as e:
        out = {"fl": fl, "Error": str(e)}
        return out

