import argparse
import copy
import glob
import json
import os
import os.path as osp
import random
import re
import shutil
import time
from datetime import datetime
from distutils.dir_util import copy_tree
from enum import Enum

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from utils.train.trainer import data_provider_solver, train, _add_arg_from_config, val_step_new
from utils.DataPrepareUtils import my_pre_transform
from utils.time_meta import record_data, print_function_runtime
from utils.utils_functions import add_parser_arguments, args2loss_fn, get_device, preprocess_config, \
    init_model_test, non_collapsing_folder


class Select(Enum):
    PERCENT = 0
    THRESHOLD = 1
    NUMBER = 3
    N_IN_PERCENT = 4
    WEIGHTED_RD = 5  # torch.utils.data.WeightedRandomSampler
    N_IN_THRESHOLD = 6


class Metric(Enum):
    EVIDENTIAL = 0
    RANDOM = 1
    ENSEMBLE = 2


class Candidate(Enum):
    N_HEAVY = 0
    BELOW_N_HEAVY = 1
    MIX_N_HEAVY = 2


class UncertaintyCal(Enum):
    STD = 0
    RANGE = 1


PUBLIC_ENUMS = {
    'Select': Select,
    'Metric': Metric,
    'Candidate': Candidate,
    'UncertaintyCal': UncertaintyCal
}


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        return None


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return d


class ALTrainer:
    """
    Active learning trainer for Frag20 dataset
    """

    def __init__(self, config_name=None, percent=None, select: Select = Select.PERCENT, threshold=None,
                 metric: Metric = Metric.EVIDENTIAL, chk=None, candidate: Candidate = Candidate.N_HEAVY, number=None,
                 init_size=None, valid_size=1000, init_n_heavy=None, action_n_heavy: str = None, fixed_valid=False,
                 skip_init_train=False, ft_lr=None, n_heavy_ratio: float = None, chk_uncertainty=False, bagging=False,
                 seed_folder=None, ext_init_folder=None, n_ensemble=None, uncertainty_cal: UncertaintyCal = None,
                 fixed_train=False, magic_i=None, explicit_init_split=None, explicit_args=None, **kwargs):
        """

        :param config_name:
        :param percent: The percentage to select in each AL cycle from candidate pool, select must be Select.PERCENT
        :param select: selection method
        :param threshold: The threshold of uncertainty to use when selecting new molecules
        :param metric: Uncertainty metric
        :param chk: Checkpoint
        :param candidate: The candidate pool to use
        :param number: The number of molecules to select in each AL cycle from candidate pool,
         select must be Select.NUMBER
        :param init_size: Initial size
        :param valid_size: validation size
        :param init_n_heavy: Initial molecule's heavy atoms
        :param action_n_heavy: AL training schedule
        :param fixed_valid: Use data_provider's valid index
        :param fixed_train: Use data_provider's train index
        :param skip_init_train: Skip the initial training
        :param ft_lr: The learning rate to use in AL
        :param n_heavy_ratio: Candidate must be Candidate.MIX_N_HEAVY. The ratio of n_heavy to < n_heavy
        :param bagging: bagging for ensemble models, currently only support n'=n
        :param seed_folder: for ensemble training only, use a pretrained ensemble folder to skip initial training
        """

        self.magic_i = magic_i
        self.fixed_train = fixed_train
        self.uncertainty_cal = uncertainty_cal
        self.ext_init_folder = ext_init_folder
        if bagging:
            assert metric == Metric.ENSEMBLE, f"Metric should be ensemble, got {metric} instead"
        if seed_folder is not None:
            assert metric == Metric.ENSEMBLE, f"Metric should be ensemble, got {metric} instead. Use skip_init_train " \
                                              f"if you are not using ensemble. "

        self.seed_folder = seed_folder
        self.bagging = bagging
        self.chk_uncertainty = chk_uncertainty
        self.n_heavy_ratio = n_heavy_ratio
        self.ft_lr = ft_lr
        self.skip_init_train = skip_init_train
        self.fixed_valid = fixed_valid
        self.action_n_heavy = action_n_heavy
        self.init_n_heavy = init_n_heavy
        self.valid_size = valid_size
        self.init_size = init_size
        self.number = number
        self.explicit_init_split = explicit_init_split

        # ----Deprecated---- #
        self.N_HEAVY_CAP = 20
        self.N_HEAVY_BOTTOM = 9
        # ----Deprecated---- #
        # ----Constants---- #
        self.N_ENSEMBLE = n_ensemble

        self.candidate = candidate
        self.config_name = config_name
        self.percent = percent
        self.select = select
        self.threshold = threshold
        self.metric = metric

        # it is used as a checkpoint: the latest n_heavy/n_cycle trained
        self.n_heavy = None
        self.n_cycle = None

        self.al_folder = chk
        self._selected_index = {}

        self._config_dict = explicit_args
        self._data_provider = None
        self._ds_n_heavy = None
        self._loss_fn = None
        self._folder_format = None
        self._n_heavy_list = None
        self._init_split = None
        self._args_processed = None

        if chk is not None:
            self.load_chk()
        else:
            prefix = self.args["folder_prefix"]
            self.al_folder = non_collapsing_folder(prefix, "_active_ALL_")

        self.prefix = osp.basename(self.args["folder_prefix"])

        if self.action_n_heavy is None:
            self.log("Please assign action_n_heavy")
        self.time_debug = self.args_processed["time_debug"]

    def train(self):
        # init distribution training
        self.setup_dist_train()
        t0 = time.time()

        # init run, can be skipped if you pretrained a model
        if self.n_cycle is None:
            if self.config_name is not None:
                shutil.copy(self.config_name, self.al_folder)
            self_info = vars(self)
            for name in self_info.keys():
                self.log(f"{name}={self_info[name]}")
            self.log(self.init_split)
            self.save_chk()

            self.log("active learning starts!")
            self.log("Init training")
            self.n_cycle = -1
            self.n_heavy = self.N_HEAVY_BOTTOM
            this_args = copy.copy(self.args)
            this_args["folder_prefix"] = self.folder_format()
            if self.skip_init_train:
                this_args["num_epochs"] = 0

            n_train = 1
            if self.metric == Metric.ENSEMBLE:
                if self.seed_folder is None:
                    n_train = self.N_ENSEMBLE
                else:
                    n_train = 0
                    init_runs = glob.glob(osp.join(self.seed_folder, "exp*_cycle_-1_*"))
                    assert len(init_runs) == self.N_ENSEMBLE, f"init runs should have {self.N_ENSEMBLE}: {init_runs}"
                    for f in init_runs:
                        # The same as: $ cp -r ${seed_folder}/exp*_cycle_-1_* ${al_folder}
                        dst_dir = osp.join(self.al_folder, osp.basename(f))
                        os.makedirs(dst_dir, exist_ok=False)
                        copy_tree(f, dst_dir)

            if self.time_debug:
                t0 = record_data("al_init_setup", t0)

            if self.seed_folder is None:
                # map multiple pretrained model to each individual ensemble model
                pretrained = glob.glob(this_args["use_trained_model"])
                pretrained.sort()
                if self.magic_i is not None:
                    pretrained = pretrained[self.magic_i: self.magic_i+1]
                assert len(pretrained) in [0, 1, n_train], f"pretrained: {pretrained}"
                if len(pretrained) == 1:
                    pretrained = pretrained * n_train
                elif len(pretrained) == 0:
                    pretrained = ["False"] * self.N_ENSEMBLE
                pretrained.sort()

                # When we only finished part of the ensemble model, we will continue at when we left
                finished = glob.glob(osp.join(self.al_folder, "exp*_cycle_-1_*"))
                if len(finished) > 0:
                    finished.sort()
                    assert len(finished) <= n_train, f"finished folders too many: {finished}"
                    last_finished = finished[-1]
                    # the last one is probably not finished
                    last_finished_arg = copy.deepcopy(this_args)
                    last_finished_arg["chk"] = last_finished
                    last_finished_arg["reset_optimizer"] = False
                    assert not self.bagging
                    train(last_finished_arg, self.data_provider, self.init_split)

                for i in range(len(finished), n_train):
                    this_pretrained = pretrained[i]
                    this_args["use_trained_model"] = this_pretrained
                    this_split = self.init_split
                    if self.bagging:
                        this_split = self.get_bagging_split(this_split)
                    meta = train(this_args, self.data_provider, this_split)
                    if self.skip_init_train:
                        for f in glob.glob(f"{this_args['use_trained_model']}/*.pt"):
                            shutil.copy(f, meta["run_directory"])
                if self.time_debug:
                    t0 = record_data("individual_runs", t0)

            self.save_chk()

        # active learning through new data selection
        for n_heavy in self.n_heavy_list[self.n_cycle + 1:]:
            self.n_heavy = n_heavy
            self.n_cycle += 1
            self.select_index()
            self.log(f"train at n cycle {self.n_cycle}, n heavy = {n_heavy}")

            # find prev trained model to load it
            prev_runs = glob.glob(f"{self.folder_format(shift=-1)}_run_*")
            if self.metric == Metric.ENSEMBLE:
                if self.n_cycle == 0 and self.seed_folder is not None:
                    # seed folders have different prefix than this one
                    prev_runs = glob.glob(osp.join(self.al_folder, "exp*_cycle_-1_*"))
                assert len(prev_runs) == self.N_ENSEMBLE, f"{prev_runs}"
            else:
                assert len(prev_runs) == 1, f"{prev_runs}"
            # folders are named and sorted by run time
            prev_runs.sort()
            current_runs = glob.glob(f"{self.folder_format(shift=0)}_run_*")
            if len(current_runs) > 0:
                assert len(current_runs) <= len(prev_runs)
                current_runs.sort()
                last_run = current_runs[-1]
                last_run_config = osp.join(last_run, "config_runtime.json")
                resume_args = self.arg_from_json(last_run_config)
                resume_args["chk"] = last_run
                this_split = self.get_current_split(self.n_cycle)
                assert not self.bagging
                train(resume_args, self.data_provider, this_split)

            for prev_run in prev_runs[len(current_runs):]:
                this_split = self.get_current_split(self.n_cycle)
                if self.bagging:
                    this_split = self.get_bagging_split(this_split)

                this_args = copy.deepcopy(self.args)
                this_args["use_trained_model"] = prev_run
                this_args["folder_prefix"] = self.folder_format()
                this_args["reset_optimizer"] = True
                if self.ft_lr is not None:
                    this_args["learning_rate"] = self.ft_lr
                this_run = glob.glob(this_args["folder_prefix"] + "*")
                if self.metric != Metric.ENSEMBLE and len(this_run) > 0:
                    self.log(f"removing folders: {this_run}")
                    for f in this_run:
                        os.makedirs("removed", exist_ok=True)
                        shutil.move(f, "removed")
                t0 = time.time()
                meta = train(this_args, self.data_provider, this_split)
                if self.time_debug:
                    t0 = record_data("individual_runs", t0)
                torch.save(this_split, osp.join(meta["run_directory"], "runtime_split.pt"))

            self.save_chk()

        if len(self.n_heavy_list) > 0:
            self.validate_simple()

        if self.time_debug:
            print_function_runtime(self.al_folder)

    def load_chk(self):
        with open(osp.join(self.al_folder, "chk.json"), "r") as f:
            self_info = json.load(f, object_hook=as_enum)
            for key in self_info.keys():
                if key not in ["al_folder", "_init_split"] and self_info[key] is not None:
                    setattr(self, key, self_info[key])
        self._selected_index = torch.load(osp.join(self.al_folder, "chk_index.pt"))

    def save_chk(self):
        t0 = time.time()
        self_info = vars(self)
        with open(osp.join(self.al_folder, "chk.json"), "w") as output:
            json.dump(self_info, output, indent=4, skipkeys=True, cls=EnumEncoder)

        torch.save(self._selected_index, osp.join(self.al_folder, "chk_index.pt"))

        if self.time_debug:
            t0 = record_data("save_chk", t0)

    def uncertainty_ensemble(self):
        self.log(f"start selecting new mols from n heavy = {self.n_heavy}")
        prev_folders = glob.glob(f"{self.folder_format(shift=-1)}_run_*")
        if self.n_cycle == 0 and self.seed_folder is not None:
            # seed folders have different prefix than this one
            prev_folders = glob.glob(osp.join(self.al_folder, "exp*_cycle_-1_*"))
        assert len(prev_folders) == self.N_ENSEMBLE
        predictions = []
        # just to avoid a warning in PyCharm
        candidate_index = None
        for prev_folder in prev_folders:
            best_model_sd = torch.load(osp.join(prev_folder, "best_model.pt"))
            model = init_model_test(self.args, best_model_sd)
            # candidate_index: the index of molecules to select from
            candidate_index = self.get_candidate(self.n_heavy, self.n_cycle)
            data_loader = DataLoader(
                self.data_provider[torch.as_tensor(candidate_index)], batch_size=self.args["valid_batch_size"],
                pin_memory=torch.cuda.is_available(), shuffle=False)
            result = val_step_new(model, data_loader, self.loss_fn, mol_lvl_detail=True, lightweight=True, config_dict=self.args)
            if result["PROP_PRED"].shape[-1] == 1:
                predictions.append(result["PROP_PRED"].view(-1, 1))
            else:
                # we have multiple targets here
                predictions.append(result["PROP_PRED"].unsqueeze(-1))
        # shape: single task: (num_mols, num_ensembles) or multi-task: (num_mols, num_targets, num_ensembles)
        predictions = torch.cat(predictions, dim=-1)
        if self.uncertainty_cal == UncertaintyCal.STD:
            uncertainty = torch.std(predictions, dim=-1, keepdim=False)
        elif self.uncertainty_cal == UncertaintyCal.RANGE:
            tmp_max, __ = torch.max(predictions, dim=-1, keepdim=False)
            tmp_min, __ = torch.min(predictions, dim=-1, keepdim=False)
            uncertainty = tmp_max - tmp_min
        else:
            raise ValueError(f"{self.uncertainty_cal} not recognized!")
        if len(uncertainty.shape) == 2:
            # choose the target with most uncertainty
            uncertainty, __ = torch.max(uncertainty, dim=-1, keepdim=False)
        else:
            assert len(uncertainty.shape) == 1, f"{uncertainty}"
        return candidate_index, uncertainty

    def uncertainty(self):
        # test on higher n_heavy and select new molecules
        self.log(f"start selecting new mols from n heavy = {self.n_heavy}")
        prev_folders = glob.glob(f"{self.folder_format(shift=-1)}_run_*")
        assert len(prev_folders) == 1
        prev_folder = prev_folders[0]
        best_model_sd = torch.load(osp.join(prev_folder, "best_model.pt"))
        model = init_model_test(self.args, best_model_sd)
        # candidate_index: the index of molecules to select from
        candidate_index = self.get_candidate(self.n_heavy, self.n_cycle)
        data_loader = DataLoader(
            self.data_provider[torch.as_tensor(candidate_index)], batch_size=self.args["valid_batch_size"],
            pin_memory=torch.cuda.is_available(), shuffle=False)
        result = val_step_new(model, data_loader, self.loss_fn, mol_lvl_detail=True, lightweight=True)
        # file too large
        # torch.save(result, osp.join(prev_folder, f"tested_on_{n_heavy}.pt"))
        if self.metric in [Metric.EVIDENTIAL]:
            assert "UNCERTAINTY" in result.keys()
            uncertainty = result["UNCERTAINTY"].view(-1)
            assert len(uncertainty) == len(candidate_index)
        else:
            uncertainty = None
        return candidate_index, uncertainty

    def select_index(self):
        t0 = time.time()
        # calculate uncertainty
        if self.metric == Metric.ENSEMBLE:
            candidate_index, uncertainty = self.uncertainty_ensemble()
        else:
            candidate_index, uncertainty = self.uncertainty()

        if self.chk_uncertainty:
            torch.save({"candidate_index": candidate_index,
                        "uncertainty": uncertainty},
                       osp.join(self.al_folder, f"uncertainty_chk_cycle{self.n_cycle}.pt"))

        # ----------------- Select new molecules according to uncertainty ------------------ #
        if self.select in [Select.PERCENT, Select.NUMBER, Select.N_IN_PERCENT, Select.WEIGHTED_RD]:
            if self.select in [Select.PERCENT, Select.N_IN_PERCENT]:
                # if it is N_IN_PERCENT:
                # select top k percent first, then randomly select n in the k percent
                n_select = int(len(candidate_index) * self.percent)
            else:
                assert self.number is not None
                n_select = self.number

            if self.metric in [Metric.EVIDENTIAL, Metric.ENSEMBLE]:
                assert isinstance(uncertainty, torch.Tensor)
                if self.select == Select.WEIGHTED_RD:
                    sampler = WeightedRandomSampler(uncertainty, n_select, replacement=False)
                    this_selected = torch.as_tensor(list(sampler))
                else:
                    this_selected = uncertainty.argsort(descending=True)[:n_select]
            else:
                # assume self.metric == "random"
                assert uncertainty is None
                perm = torch.randperm(len(candidate_index))
                this_selected = perm[:n_select]

            if self.select == Select.N_IN_PERCENT and self.number < n_select:
                # this_selected: top k selected molecules
                # we then randomly select n in this_selected
                # if k percent molecules is fewer than n, do nothing.
                perm = torch.randperm(this_selected.shape[-1])
                this_selected = this_selected[perm[:self.number]]

        elif self.select in [Select.THRESHOLD, Select.N_IN_THRESHOLD]:
            assert isinstance(uncertainty, torch.Tensor), f"{uncertainty}"
            uncertainty: torch.Tensor = uncertainty.view(-1)
            this_selected = (uncertainty > self.threshold).nonzero().view(-1)
            self.log(f"cycle: {self.n_cycle}, selecting based on threshold {self.threshold},"
                     f" {uncertainty.shape[-1]} candidate(s), {this_selected.shape[-1]} are bigger than threshold")
            if self.select == Select.N_IN_THRESHOLD and this_selected.shape[-1] > self.number:
                perm = torch.randperm(this_selected.shape[-1])
                this_selected = this_selected[perm[:self.number]]
                self.log(f"Only {this_selected.shape[-1]} are randomly selected")
        else:
            raise ValueError
        select_in_dataset = candidate_index[this_selected]
        self.log(f"{len(select_in_dataset)} in {len(uncertainty)} are selected.")
        self.add_selected(self.n_cycle, select_in_dataset)

        if self.time_debug:
            t0 = record_data("select_index", t0)

    def get_candidate(self, n_heavy, n_cycle, is_training=True, candidate=None):
        if candidate is None:
            candidate = self.candidate
        if is_training:
            if n_cycle >= 0:
                assert max(list(self._selected_index.keys())) == n_cycle - 1
            else:
                assert len(list(self._selected_index.keys())) == 0

        if candidate == Candidate.N_HEAVY:
            b4_rm = self.get_index_heavy(n_heavy, no_large=False)
        elif candidate in [Candidate.BELOW_N_HEAVY, Candidate.MIX_N_HEAVY]:
            b4_rm = self.get_index_heavy(n_heavy, no_large=True)
        else:
            raise ValueError

        set_b4_rm = set(b4_rm.tolist())
        index_current = torch.cat([self._selected_index[key] for key in range(-1, n_cycle)])
        set_current = set(index_current.tolist())
        set_candidate_index = set_b4_rm.difference(set_current)

        if candidate in [Candidate.BELOW_N_HEAVY, Candidate.MIX_N_HEAVY]:
            assert len(set_b4_rm) == len(set_candidate_index) + len(set_current)

        # Extra steps for mixed n_heavy and below_n_heavy
        # Includes all n_heavy molecules and part of below_n_heavy molecules depending on n_heavy_ratio
        if candidate == Candidate.MIX_N_HEAVY:
            set_n_heavy = set(self.get_index_heavy(n_heavy, no_large=False).tolist())
            set_candidate_n_heavy = set_candidate_index.intersection(set_n_heavy)
            set_candidate_below_n_heavy = set_candidate_index.difference(set_n_heavy)
            n_below = int(len(set_candidate_n_heavy) / self.n_heavy_ratio)
            set_below_sampled = random.sample(set_candidate_below_n_heavy, n_below)
            set_candidate_index = set_candidate_n_heavy.union(set_below_sampled)

        return torch.as_tensor(list(set_candidate_index)).long()

    def get_current_split(self, n_cycle=None):
        keys = list(self._selected_index.keys())
        if n_cycle is not None:
            assert max(keys) == n_cycle
        train_index = torch.cat([self._selected_index[key] for key in self._selected_index.keys()])
        return self.add_valid(train_index)

    def add_selected(self, n_cycle, index: torch.Tensor):
        # add selected index to the training set.
        assert isinstance(n_cycle, int)
        assert n_cycle not in self._selected_index.keys()
        self._selected_index[n_cycle] = index

    def get_index_heavy(self, n_heavy, remove_valid=True, no_large=True):
        # get the index of molecules that is equal or smaller than n_heavy in the dataset
        # removes pre-defined valid if possible
        n_heavy = str(n_heavy)
        if "-" in n_heavy:
            # a-b
            tmp = n_heavy.split("-")
            assert len(tmp) == 2, tmp
            n_min = int(tmp[0])
            n_max = int(tmp[1])
            idx = ((self.ds_n_heavy <= n_max) & (self.ds_n_heavy >= n_min)).nonzero().view(-1)
        else:
            n_heavy = int(n_heavy)
            if no_large:
                idx = (self.ds_n_heavy <= n_heavy).nonzero().view(-1)
            else:
                idx = (self.ds_n_heavy == n_heavy).nonzero().view(-1)
        if remove_valid:
            set_idx = set(idx.tolist())
            set_valid = set(self.data_provider.val_index.tolist())
            set_idx_rm = set_idx.difference(set_valid)
            if self.data_provider.test_index is not None:
                set_test = set(self.data_provider.test_index.tolist())
                set_idx_rm = set_idx_rm.difference(set_test)
            idx = torch.as_tensor(list(set_idx_rm)).long()
        return idx

    def log(self, msg, end="\n", time=True, p=False):
        assert self.al_folder is not None
        msg = str(msg)
        if time:
            current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            msg = current_time + ": " + msg
        with open(osp.join(self.al_folder, "al_log.txt"), "a") as f:
            f.write(msg + end)
        if p:
            print(msg)

    def setup_dist_train(self):
        self.args["local_rank"] = int(os.environ.get("LOCAL_RANK") if os.environ.get("LOCAL_RANK") is not None else 0)
        self.args["is_dist"] = (os.environ.get("RANK") is not None)

    def validate_simple(self):
        # print more meta data to validation / debug
        t0 = time.time()

        self.log("------------------------")
        for n_cycle in self._selected_index:
            if n_cycle <= 0:
                continue
            n_heavy = self.n_heavy_list[n_cycle]
            selected_heavy: torch.Tensor = self.ds_n_heavy[self._selected_index[n_cycle]]
            if n_heavy == self.N_HEAVY_BOTTOM:
                num_true = (selected_heavy <= n_heavy).sum()
                num_false = (selected_heavy > n_heavy).sum()
            else:
                num_true = (selected_heavy == n_heavy).sum()
                num_false = (selected_heavy != n_heavy).sum()
            self.log(f"selected n heavy: {n_heavy}. Number mol not with n_heavy: {num_false}."
                     f" Number mol with n_heavy: {num_true}.")
            self.log(
                f"cycle={n_cycle}, hist: {np.histogram(selected_heavy.numpy(), list(range(1, self.N_HEAVY_CAP + 2)))}")

        self.log("---- All training set ----")
        final_split = self.get_current_split()
        for split in ["train_index", "valid_index", "test_index"]:
            this_index = final_split[split]
            if this_index is not None:
                selected_heavy: torch.Tensor = self.ds_n_heavy[this_index]
                self.log(f"TOTAL {split},"
                         f" hist: {np.histogram(selected_heavy.numpy(), list(range(1, self.N_HEAVY_CAP + 2)))}")

        if self.time_debug:
            t0 = record_data("validate_simple", t0)

    def validate_selection_cycle(self, n_cycle, save=True, print_fig=False):
        # rerun the selection_index to validate the selection process
        # on HPC: save True, print_fig False
        # local: save False, print_fig True
        def find(t: torch.Tensor, values, reverse=False):
            if reverse:
                return torch.nonzero(t.unsqueeze(-1) != values)[:, 0]
            else:
                return torch.nonzero(t.unsqueeze(-1) == values)[:, 0]

        if self.metric == Metric.ENSEMBLE:
            if save:
                ens_folders = glob.glob(osp.join(self.al_folder, f"exp*_cycle_{n_cycle - 1}_run_*"))
                assert len(ens_folders) == self.N_ENSEMBLE, f"ens_folders: {ens_folders}"
                this_candidate = Candidate.BELOW_N_HEAVY if self.candidate == Candidate.MIX_N_HEAVY else None
                candidate_index = self.get_candidate(self.n_heavy_list[n_cycle], n_cycle, is_training=False,
                                                     candidate=this_candidate)

                predictions = []
                tgt = None
                selected_idx_in_candidate = find(candidate_index, self._selected_index[n_cycle])
                others_in_candidate = set(list(range(len(candidate_index)))).difference(
                    set(selected_idx_in_candidate.tolist()))
                others_in_candidate = torch.as_tensor(list(others_in_candidate))
                assert len(selected_idx_in_candidate) + len(others_in_candidate) == len(candidate_index), \
                    f"{len(selected_idx_in_candidate)}, {len(others_in_candidate)}, {len(candidate_index)},"
                for prev_folder in ens_folders:
                    best_model_sd = torch.load(osp.join(prev_folder, "best_model.pt"), map_location=get_device())
                    model = init_model_test(self.args, best_model_sd)
                    data_loader = DataLoader(
                        self.data_provider[torch.as_tensor(candidate_index)], batch_size=self.args["valid_batch_size"],
                        pin_memory=torch.cuda.is_available(), shuffle=False)
                    result = val_step_new(model, data_loader, self.loss_fn, mol_lvl_detail=True, lightweight=True)
                    assert result["PROP_PRED"].shape[-1] == 1
                    predictions.append(result["PROP_PRED"].view(-1, 1))
                    if tgt is None:
                        tgt = result["PROP_TGT"].view(-1)

                predictions = torch.cat(predictions, dim=-1)
                print(f"{len(selected_idx_in_candidate)} / {len(self._selected_index[n_cycle])}")

                meta = {"predictions": predictions,
                        "selected_idx_in_candidate": selected_idx_in_candidate, "tgt": tgt,
                        "others_in_candidate": others_in_candidate}
                torch.save(meta, osp.join(self.al_folder, f"validate_cycle_{n_cycle}_meta.pt"))

            if print_fig:
                meta = torch.load(osp.join(self.al_folder, f"validate_cycle_{n_cycle}_meta.pt"))
                selected_idx_in_candidate = meta["selected_idx_in_candidate"]
                others_in_candidate = meta["others_in_candidate"]
                if not isinstance(others_in_candidate, torch.Tensor):
                    others_in_candidate = torch.as_tensor(list(others_in_candidate))
                tgt = meta["tgt"]
                predictions = meta["predictions"]
                from src.uncertainty_curves import uncertainty_curves_ens
                uncertainty_curves_ens(predictions, tgt, self.al_folder, n_cycle,
                                       self.N_ENSEMBLE, selected_idx_in_candidate, others_in_candidate)

    def sample_random(self):
        self.log(self.init_split)
        for n_cycle, n_heavy in enumerate(self.n_heavy_list):
            candidate = self.get_candidate(n_heavy, n_cycle)
            if self.select == Select.PERCENT:
                n_select = len(candidate) * self.percent
                n_select = int(n_select)
            elif self.select in [Select.NUMBER, Select.N_IN_PERCENT, Select.WEIGHTED_RD, Select.N_IN_THRESHOLD]:
                n_select = self.number
            else:
                raise ValueError(f"{self.select}")
            perm = torch.randperm(len(candidate))
            selected = candidate[perm[:n_select]]
            self.add_selected(n_cycle, selected)
        self.save_chk()

    @property
    def args(self):
        if self._config_dict is None:
            self._config_dict = self.arg_from_file(self.config_name)
        return self._config_dict

    @staticmethod
    def arg_from_file(f_path):
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        parser = add_parser_arguments(parser)

        # parse config file
        args, unknown = parser.parse_known_args(["@" + f_path])
        args.config_name = f_path
        return vars(args)

    @staticmethod
    def arg_from_json(f_path):
        with open(f_path) as f:
            return json.load(f)

    @property
    def args_processed(self):
        if self._args_processed is None:
            self._args_processed = preprocess_config(self.args)
        return self._args_processed

    @property
    def data_provider(self) -> InMemoryDataset:
        if self._data_provider is None:
            default_kwargs = {'data_root': self.args["data_root"], 'pre_transform': my_pre_transform,
                              'record_long_range': True, 'type_3_body': 'B', 'cal_3body_term': True}
            data_provider_class, _kwargs = data_provider_solver(self.args, default_kwargs)
            _kwargs = _add_arg_from_config(_kwargs, self.args)
            self._data_provider: InMemoryDataset = data_provider_class(**_kwargs)
        return self._data_provider

    @property
    def ds_n_heavy(self) -> torch.Tensor:
        # number of heavy atoms in the dataset
        # this is required to select molecules according to heavy atoms
        if self._ds_n_heavy is None:
            tmp_n_heavy = []
            for i in range(len(self.data_provider)):
                this_data = self.data_provider[i]
                tmp_n_heavy.append((this_data.Z != 1).sum())
            self._ds_n_heavy = torch.as_tensor(tmp_n_heavy).view(-1).long()
        return self._ds_n_heavy

    @property
    def init_split(self) -> dict:
        if self._init_split is None:
            if self.ext_init_folder is not None:
                assert not self.fixed_train
                # use external folder's checkpoint as initialization
                ext_idx_chk = osp.join(self.ext_init_folder, "chk_index.pt")
                ext_idx_chk = glob.glob(ext_idx_chk)
                assert len(ext_idx_chk) == 1, f"ext_idx_chk: {ext_idx_chk}"
                ext_idx_chk = ext_idx_chk[0]
                ext_idx_chk = torch.load(ext_idx_chk)
                init_train = torch.cat([ext_idx_chk[key] for key in ext_idx_chk.keys()])
                self._init_split = self.add_valid(init_train)
            elif self.explicit_init_split is not None:
                self._init_split = self.explicit_init_split
                self.add_selected(-1, self.explicit_init_split["train_index"])
            else:
                if self.fixed_train:
                    train_index = torch.as_tensor(self.data_provider.train_index).view(-1)
                else:
                    if self.init_n_heavy is None:
                        train_index = self.get_index_heavy(self.N_HEAVY_BOTTOM, no_large=True)
                    else:
                        train_index = self.get_index_heavy(self.init_n_heavy, no_large=True)
                if self.init_size is not None:
                    assert isinstance(self.init_size, int)
                    # init split should be the same
                    np.random.seed(19260817)
                    perm = np.random.permutation(len(train_index))
                    perm = torch.as_tensor(perm).long()
                    train_index = train_index[perm[:self.init_size]]
                self._init_split = self.add_valid(train_index)
                if -1 in self._selected_index.keys():
                    assert (self._selected_index[-1] == train_index).sum() == train_index.shape[0]
                else:
                    self.add_selected(-1, train_index)
        return self._init_split

    @property
    def loss_fn(self):
        if self._loss_fn is None:
            self._loss_fn = args2loss_fn(self.args)
        return self._loss_fn

    @property
    def n_heavy_list(self):
        if self._n_heavy_list is None:
            if self.action_n_heavy is None:
                raise ValueError("action_n_heavy should not be None")
                # self._n_heavy_list = range(self.n_heavy + 1, self.N_HEAVY_CAP + 1)
            else:
                self._n_heavy_list = self.action_n_heavy.split()
        return self._n_heavy_list

    def folder_format(self, shift=0, n_cycle=None) -> str:
        if self.action_n_heavy is None:
            return f"{self.al_folder}/{self.prefix}_heavy_{self.n_heavy + shift}"
        else:
            val = self.n_cycle + shift if n_cycle is None else n_cycle
            return f"{self.al_folder}/{self.prefix}_cycle_{val}"

    def add_valid(self, index):
        if self.fixed_valid:
            self.log(f"using fixed valid split, ignoring valid size = {self.valid_size}")
            train_index = index
            valid_index = self.data_provider.val_index
            set_train_index = set(train_index.tolist())
            set_valid_index = set(valid_index.tolist())
            assert len(set_train_index.intersection(set_valid_index)) == 0
            result = {
                "train_index": train_index,
                "valid_index": valid_index,
                "test_index": self.data_provider.test_index
            }
        else:
            perm = torch.randperm(len(index))
            train_size = len(index) - self.valid_size
            assert train_size > 0
            result = {
                "train_index": index[perm[:train_size]],
                "valid_index": index[perm[-self.valid_size:]],
                "test_index": self.data_provider.test_index
            }
        return result

    @staticmethod
    def get_bagging_split(split):
        # I don't want to change the initial values
        result = copy.deepcopy(split)

        n = split["train_index"].shape[0]
        n_prime = n
        bagging_index = torch.randint(low=0, high=n, size=(n_prime,))
        result["train_index"] = result["train_index"][bagging_index]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", default="config-al-small.txt")
    parser.add_argument("--percent", default=None, type=float)
    parser.add_argument("--chk", type=str, default=None)
    parser.add_argument("--select", type=str, default="PERCENT")
    parser.add_argument("--metric", type=str, default="EVIDENTIAL")
    parser.add_argument("--candidate", type=str, default="N_HEAVY")
    parser.add_argument("--uncertainty_cal", type=str, default="STD")
    parser.add_argument("--number", default=None, type=int)
    parser.add_argument("--init_size", default=None, type=int)
    parser.add_argument("--init_n_heavy", default=None, type=int)
    parser.add_argument("--ext_init_folder", default=None, type=str)
    parser.add_argument("--action_n_heavy", default=None, type=str)
    parser.add_argument("--valid_size", default=1000, type=int)
    parser.add_argument("--fixed_valid", action="store_true")
    parser.add_argument("--fixed_train", action="store_true")
    parser.add_argument("--bagging", action="store_true")
    parser.add_argument("--skip_init_train", action="store_true")
    parser.add_argument("--seed_folder", default=None, type=str)
    parser.add_argument("--ft_lr", default=None, type=float)
    parser.add_argument("--n_heavy_ratio", default=None, type=float)
    parser.add_argument("--threshold", default=None, type=float,
                        help="Selection threshold, note it is in eV for ensemble models")
    parser.add_argument("--n_ensemble", default=5, type=int)

    parser.add_argument("--magic_i", default=None, help="Do not use it, temp variable for personal use.", type=int)

    parser.add_argument("--sample_rd", action="store_true")
    args = vars(parser.parse_args())

    # Convert Camel to Snake
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    for key in PUBLIC_ENUMS.keys():
        key_lower = pattern.sub('_', key).lower()
        if key_lower in args.keys():
            args[key_lower] = getattr(PUBLIC_ENUMS[key], args[key_lower])

    trainer = ALTrainer(**args)
    if args["sample_rd"]:
        trainer.sample_random()
    else:
        trainer.train()


if __name__ == '__main__':
    # tmp = ALTrainer(chk="../raw_data/exp200-400/exp342_active_ALL_2021-08-25_145756")
    main()
