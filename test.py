from utils.utils_functions import torchdrug_imports
torchdrug_imports()

import argparse
import glob

from utils.eval.tester import Tester
from utils.eval.mdn_extractor import MDNExtractor
from utils.eval.mdn_embed_tester import MDNEmbedTester


def test_folder(folder_name, **kwargs):
    tester_cls = None
    if kwargs["gen_mdn_embed"]:
        tester_cls = MDNExtractor
    if kwargs["mdn_embed_test"]:
        assert tester_cls is None, tester_cls
        tester_cls = MDNEmbedTester
    del kwargs["gen_mdn_embed"]

    assert tester_cls is None, tester_cls
    tester_cls = Tester
    tester = tester_cls(folder_name=folder_name, **kwargs)
    tester.run_test()


def test_all():
    parser = argparse.ArgumentParser()
    # parser = add_parser_arguments(parser)
    parser.add_argument('--folder_names', type=str)
    parser.add_argument('--config_folders', default=None, type=str)
    parser.add_argument("--explicit_ds_config", default=None, type=str,
                        help="Explicitly specify dataset config, used when testing on an external dataset.")
    parser.add_argument("--overwrite_args", default=None, help="A json file overwriting model configs.")
    parser.add_argument('--x_forward', default=0, type=int)
    parser.add_argument('--n_forward', default=5, type=int)
    parser.add_argument('--use_exist', action="store_true")
    parser.add_argument('--include_train', action="store_true")
    parser.add_argument('--lightweight', action="store_true")
    parser.add_argument("--no_runtime_split", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--use_tqdm", action="store_true")

    # ------------ MDN Embedding ------------ #
    parser.add_argument("--gen_mdn_embed", action="store_true", help="Instead of normal testing, generate MDN embedding for the dataset.")
    parser.add_argument("--prot_seq_info", default=None, help="sequence information of proteins which is needed to generate MDN embeddings.")
    parser.add_argument("--embed_save_f", default=None, help="Output file of the computed embedding.")
    # ------------ MDN Embedding Test ----------- #
    parser.add_argument("--mdn_embed_test", action="store_true", help="Test MDN embedding models")
    # external MDN scores
    parser.add_argument("--compute_external_mdn", action="store_true")
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
