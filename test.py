from utils.utils_functions import torchdrug_imports
torchdrug_imports()

import argparse

from utils.eval.tester import Tester


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str)
    parser.add_argument("--explicit_ds_config", default=None, type=str,
                        help="Explicitly specify dataset config, used when testing on an external dataset.")
    parser.add_argument("--overwrite_args", default=None, help="A json file overwriting model configs.")
    parser.add_argument('--use_exist', action="store_true")
    parser.add_argument("--no_runtime_split", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--compute_external_mdn", action="store_true")
    parser.add_argument("--no_pkd_score", action="store_true", help="Only predicts NMDN score, ingnoring pKd score.")
    parser.add_argument("--diffdock_nmdn_result", default=None, type=str, action="append")
    parser.add_argument("--linf9_csv", default=None, type=str, help="RMSD information")
    args = parser.parse_args()

    print('testing folder: {}'.format(args.folder_name))
    args = vars(args)
    tester = Tester(**args)
    tester.run_test()


if __name__ == "__main__":
    run_test()
