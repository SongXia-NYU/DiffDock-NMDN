"""
Merge independent runs into a single folder for testing
"""
import argparse
import os
import os.path as osp
import shutil
from glob import glob
import json


def merge_runs(root, exp_id, test):
    os.makedirs(osp.join(root, "tmp"), exist_ok=True)
    folders = glob(osp.join(root, f"exp{exp_id}_active_ALL_*"))
    folders.sort()

    for f in folders:
        shutil.copytree(f, osp.join(root, "tmp", osp.basename(f)), dirs_exist_ok=True)

    for f in folders[1:]:
        shutil.move(glob(osp.join(f, "exp*_cycle_-1_*"))[0], folders[0])
        shutil.rmtree(f)

    if test:
        from generate_test_al import generate_test_al
        generate_test_al(str_id=str(exp_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="..")
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    args = vars(args)

    merge_runs(**args)


if __name__ == '__main__':
    main()
