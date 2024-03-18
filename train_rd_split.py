import copy
import os.path as osp
from datetime import datetime

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from utils.data.DummyIMDataset import DummyIMDataset
from active_learning import Metric, ALTrainer
from utils.train.trainer import flex_parse, train, dataset_from_args
from utils.utils_functions import non_collapsing_folder


def main():
    args = flex_parse(add_extra_args=add_extra_args)
    data_provider = dataset_from_args(args)
    assert isinstance(data_provider, DummyIMDataset), data_provider

    folder_prefix = args["folder_prefix"]
    root = non_collapsing_folder(folder_prefix, identify="_RDrun_")
    args["folder_prefix"] = f"{root}/{folder_prefix}"

    magic_i = args["magic_i"]
    for i in range(args["n_runs"]):
        split_f = ""
        if args["rd_split_folder"] is None:
            split = get_rd_split(len(data_provider), **args)
        else:
            split_f = osp.join(args["rd_split_folder"], f"split_{i+magic_i}.pt")
            print(f"Using split file: {split_f}")
            split = torch.load(split_f)

        if args["n_ens"] == 1:
            train(args, explicit_split=split, data_provider=data_provider)
        else:
            # Train ensemble model
            # Do not want to change in-place
            args_copy = copy.deepcopy(args)
            ens_args = {
                "n_ensemble": args["n_ens"], "metric": Metric.ENSEMBLE, "action_n_heavy": "", "fixed_train": True,
                "fixed_valid": True, "explicit_init_split": split, "explicit_args": args_copy
            }
            ens_trainer = ALTrainer(**ens_args)
            ens_trainer.log(f"Using split file: {split_f}")
            ens_trainer.train()


def get_rd_split(size, split, explicit_train_size=None, freeopen_special=False, freeopen_special_better=False,
                 freeopen_special_1=False, **kwargs):
    if freeopen_special or freeopen_special_better or freeopen_special_1:
        assert split == "811"
        real_size = size
        assert real_size == 14339, f"Dataset might be wrong: {real_size}"
        # special treatment for freesolv_openchem dataset: use the first 639 for testing and validation
        # use the rest for training
        size = 639

    # 80/10/10 split
    if split == "811":
        test_size = size // 10
        valid_size = test_size
    elif split == "openchem_logP":
        test_size = size // 5
        valid_size = 1000
    else:
        raise ValueError(f"Invalid split: {split}")
    index_array = np.arange(size)
    train_valid_split, test_split = train_test_split(index_array, test_size=test_size)
    train_split, valid_split = train_test_split(train_valid_split, test_size=valid_size)
    split = {
        "train_index": torch.as_tensor(train_split),
        "valid_index": torch.as_tensor(valid_split),
        "test_index": torch.as_tensor(test_split)
    }
    if explicit_train_size is not None:
        split["train_index"] = split["train_index"][:explicit_train_size]
    if freeopen_special:
        assert not freeopen_special_better
        split["train_index"] = torch.concat([split["train_index"], torch.arange(size, real_size)])
    if freeopen_special_better or freeopen_special_1:
        assert not freeopen_special
        openchem_index = torch.arange(size, real_size)
        # testing for openchem
        openchem_train, openchem_test = train_test_split(openchem_index, test_size=real_size // 5 - test_size)
        if freeopen_special_1:
            openchem_train, openchem_valid = train_test_split(openchem_train, test_size=1000 - valid_size)
            split["valid_index"] = torch.concat([split["valid_index"], openchem_valid])
        split["train_index"] = torch.concat([split["train_index"], openchem_train])
        split["test_index"] = torch.concat([split["test_index"], openchem_test])
    return split


def add_extra_args(parser):
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--split", type=str, default="811")
    parser.add_argument("--explicit_train_size", type=int, default=None)
    parser.add_argument("--freeopen_special", action="store_true", help="a special split for the combined dataset")
    parser.add_argument("--freeopen_special_better", action="store_true",
                        help="a special split for testing both freesolv and openchem")
    parser.add_argument("--freeopen_special_1", action="store_true", help="1000 valid size")
    parser.add_argument("--n_ens", type=int, default=1)
    parser.add_argument("--rd_split_folder", default=None)
    parser.add_argument("--magic_i", type=int, default=None)
    return parser


if __name__ == '__main__':
    main()
