import functools
import os
import os.path as osp
import yaml
import json

from rdkit.Chem import AllChem

from utils.utils_functions import time_limit



def solv_num_workers():
    try:
        n_cpu_avail = len(os.sched_getaffinity(0))
    except AttributeError as e:
        print(f"WARNING: {e}")
        n_cpu_avail = None
    n_cpu = os.cpu_count()
    num_workers = n_cpu_avail if n_cpu_avail is not None else n_cpu
    return n_cpu_avail, n_cpu, num_workers


def cutoff_parser(cutoff_ligand, cutoff_pl, cutoff_protein):
    name = f"{cutoff_ligand}${cutoff_pl}${cutoff_protein}"
    mp_kwargs = {"cutoff_ligand": cutoff_ligand, "cutoff_pl": cutoff_pl, "cutoff_protein": cutoff_protein}
    return name, mp_kwargs


def adfr_solver():
    if osp.exists("/scratch/sx801/"):
        SCRIPT_ROOT = "/scratch/sx801/scripts"
    else:
        SCRIPT_ROOT = "/home/carrot_of_rivia/Documents/PycharmProjects"
    PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor"
    PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_ligand"
    return PREPARE_PROT, PREPARE_LIG

# https://github.com/faif/python-patterns/blob/master/patterns/creational/lazy_evaluation.py
def cached_dict(cache_path: str):
    if not osp.exists(cache_path):
        return lambda fn: fn

    if cache_path.endswith(".yaml"):
        with open(cache_path) as f:
            d = yaml.safe_load(f)
    elif cache_path.endswith(".json"):
        with open(cache_path) as f:
            d = json.load(f)
    else: raise ValueError("Cannot recognize file: " + cache_path)

    def _inner(*args, **kwargs):
        return d
    return lambda fn: _inner

def ff_optimize(mol, ids, timeout: int = 10):
    with time_limit(timeout):
        ### Check MMFF parms ###
        if AllChem.MMFFHasAllMoleculeParams(mol):
            ### MMFF optimize ###
            method = "MMFF"
            for cid in ids:
                _ = AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        else:
            ### UFF optimize ###
            method = "UFF"
            for cid in ids:
                _ = AllChem.UFFOptimizeMolecule(mol, confId=cid)
    return method


if __name__ == "__main__":
    pass
