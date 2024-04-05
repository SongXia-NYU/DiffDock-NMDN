from utils.utils_functions import torchdrug_imports
torchdrug_imports()

import argparse
import glob

from utils.eval.tester import Tester
from utils.eval.mdn_extractor import MDNExtractor
from utils.eval.mdn_embed_tester import MDNEmbedTester


def test_folder(folder_name, **kwargs):
    for key in list(kwargs.keys()):
        if kwargs[key] == "#remove_if_not_default":
            del kwargs[key]
    if kwargs["diffdock_nmdn_result"] is None:
        del kwargs["diffdock_nmdn_result"]
    tester = Tester(folder_name=folder_name, **kwargs)
    tester.run_test()


def test_all():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_names', type=str)
    parser.add_argument('--config_folders', default=None, type=str)
    parser.add_argument("--explicit_ds_config", default=None, type=str,
                        help="Explicitly specify dataset config, used when testing on an external dataset.")
    parser.add_argument("--overwrite_args", default=None, help="A json file overwriting model configs.")
    parser.add_argument('--use_exist', action="store_true")
    parser.add_argument('--include_train', action="store_true")
    parser.add_argument("--no_runtime_split", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument("--compute_external_mdn", action="store_true")
    parser.add_argument("--no_pkd_score", action="store_true", help="Only predicts NMDN score, ingnoring pKd score.")
    parser.add_argument("--diffdock_nmdn_result", default=None, type=str, action="append")
    _args = parser.parse_args()

    run_dirs = glob.glob(_args.folder_names)

    if _args.config_folders is not None:
        config_folders = glob.glob(_args.config_folders)
        assert len(config_folders) == 1
        config_folder = config_folders[0]
    else:
        config_folder = None

    kwargs = vars(_args)
    # a hell of logic
    kwargs["ignore_train"] = not _args.include_train
    del kwargs["folder_names"]
    del kwargs["config_folders"]
    del kwargs["include_train"]
    kwargs["config_folder"] = config_folder

    for name in run_dirs:
        print('testing folder: {}'.format(name))
        test_folder(name, **kwargs)


if __name__ == "__main__":
    test_all()
