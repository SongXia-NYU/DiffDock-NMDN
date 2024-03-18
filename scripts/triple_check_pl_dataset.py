from utils.train.trainer import dataset_from_args
from glob import glob
import os.path as osp
import argparse
from tqdm import tqdm
from utils.utils_functions import add_parser_arguments
"""
Triple check to make sure that the training set does not contain CASF proteins.
"""

def triple_check():
    casf_folders = glob("/vast/sx801/CASF-2016-cyang/coreset/*")
    casf_pdbs = [osp.basename(p) for p in casf_folders]
    casf_pdbs = set(casf_pdbs)
    print(casf_pdbs)
    assert len(casf_pdbs) == 285

    config_name = "/scratch/sx801/scripts/physnet-dimenet1/MartiniDock/results/exp_pl_206_run_2023-02-06_163034__578983/config-exp_pl_206.txt"
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    args, unknown = parser.parse_known_args(["@" + config_name])
    args.config_name = config_name
    args = vars(args)
    ds = dataset_from_args(args)
    for split_name in ["train_index", "val_index"]:
        print((split_name))
        for i in tqdm(getattr(ds, split_name)):
            this_pdb = ds.idx2pdb_mapper[i.item()]
            if this_pdb in casf_pdbs:
                print(this_pdb)


if __name__ == "__main__":
    triple_check()
