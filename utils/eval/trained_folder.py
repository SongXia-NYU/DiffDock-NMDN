import argparse
import copy
import glob
import logging
import os
import os.path as osp
from datetime import datetime
import tqdm

import torch
from torch.optim.swa_utils import update_bn
from torch_geometric.loader import DataLoader

from utils.DataPrepareUtils import my_pre_transform
from Networks.PhysDimeNet import PhysDimeNet
from Networks.UncertaintyLayers.swag import SWAG
from utils.train.trainer import data_provider_solver, _add_arg_from_config, remove_extra_keys
from utils.utils_functions import args2loss_fn, remove_handler, get_device, floating_type, init_model_test, solv_num_workers, validate_index, \
    add_parser_arguments, preprocess_config


class TrainedFolder:
    """
    Load a trained folder for performance evaluation and other purposes.
    A trained folder is a folder generated by train.py. For example: 'exp_pl_005_run_2022-06-03_164656__676997'
    """
    def __init__(self, folder_name, config_folder=None):
        self.config_folder = config_folder
        self.folder_name = folder_name

        self._logger = None
        self._test_dir = None
        self._args = None
        self._args_raw = None
        self._data_provider = None
        self._data_provider_test = None
        self._config_name = None
        self._loss_fn = None
        self._model = None

        self._ds_options = None
        self._ds_cls = None
        self._num_workers = None

        self.mae_fn = torch.nn.L1Loss(reduction='mean')
        self.mse_fn = torch.nn.MSELoss(reduction='mean')

    def update_bn(self):
        train_dl = DataLoader(self.ds[torch.as_tensor(self.ds.train_index)], batch_size=self.args["valid_batch_size"],
                             shuffle=False, num_workers=self.num_workers)
        update_bn(tqdm.tqdm(train_dl), self.model, device=get_device())
        torch.save(self.model.state_dict(), osp.join(self.folder_name, "best_model_bn_updated.pt"))

    def info(self, msg: str):
        if self.save_root is None:
            return
            
        self.logger.info(msg)

    @property
    def model(self):
        if self._model is None:
            use_swag = (self.args["uncertainty_modify"].split('_')[0] == 'swag')
            if use_swag:
                net = PhysDimeNet(cfg=self.args, **self.args)
                net = net.to(get_device())
                net = net.type(floating_type)
                net = SWAG(net, no_cov_mat=False, max_num_models=20)
                model_data = torch.load(os.path.join(self.folder_name, 'swag_model.pt'), map_location=get_device())
                net.load_state_dict(model_data)
            else:
                model_data = torch.load(os.path.join(self.folder_name, 'best_model.pt'), map_location=get_device())
                # temp fix, to be removed
                # model_data = fix_model_keys(model_data)
                net = init_model_test(self.args, model_data, None)
            self._model = net
        return self._model

    @property
    def num_workers(self):
        if self._num_workers is None:
            if not torch.cuda.is_available():
                # I use cpu to debug. Setting num_worker to 0 helps me to debug the data_loader
                self._num_workers = 0
                return self._num_workers

            n_cpu_avail, n_cpu, num_workers = solv_num_workers()
            if self.args["proc_in_gpu"]:
                num_workers = 0
                self.info("Since data will be preprocessed into GPU, num_workers will be set to 0")
            self.info(f"Number of total CPU: {n_cpu}")
            self.info(f"Number of available CPU: {n_cpu_avail}")
            self.info(f"Number of workers used: {num_workers}")
            self._num_workers = num_workers
        return self._num_workers

    @property
    def loss_fn(self):
        if self._loss_fn is None:
            self._loss_fn = args2loss_fn(self.args)
        return self._loss_fn

    @property
    def args(self):
        if self._args is None:
            _args = copy.deepcopy(self.args_raw)

            if _args["ext_atom_features"] is not None:
                # infer the dimension of external atom feature
                ext_atom_feature = getattr(self.ds[[0]].data, _args["ext_atom_features"])
                ext_atom_dim = ext_atom_feature.shape[-1]
                _args["ext_atom_dim"] = ext_atom_dim
                del ext_atom_feature

            inferred_prefix = self.folder_name.split('_run_')[0]
            if _args["folder_prefix"] != inferred_prefix:
                print('overwriting folder {} ----> {}'.format(_args["folder_prefix"], inferred_prefix))
                _args["folder_prefix"] = inferred_prefix
            _args["requires_atom_prop"] = True

            self._args = _args
        return self._args

    @property
    def ds(self):
        if self._data_provider is None:
            _data_provider = ds_from_args(self.args_raw, rm_keys=False)

            # The lines below are dealing with the logic that I separate some test set from training set into
            # different files, which makes the code messy. It is not used in my relatively new datasets.
            if isinstance(_data_provider, tuple):
                _data_provider_test = _data_provider[1]
                _data_provider = _data_provider[0]
            else:
                _data_provider_test = _data_provider
            self._data_provider = _data_provider
            self._data_provider_test = _data_provider_test
        return self._data_provider

    @property
    def ds_test(self):
        if self._data_provider_test is None:
            # it was inited in self.data_provider
            __ = self.ds
        return self._data_provider_test

    @property
    def args_raw(self):
        if self._args_raw is None:
            if self.config_folder is not None:
                _args_raw, _config_name = read_folder_config(self.config_folder)
            else:
                _args_raw, _config_name = read_folder_config(self.folder_name)
            self._args_raw = _args_raw
            self._config_name = _config_name
        return self._args_raw

    @property
    def config_name(self):
        if self._config_name is None:
            __ = self.args_raw
        return self._config_name

    @property
    def save_root(self):
        return None

    @property
    def logger(self):
        if self._logger is None:
            remove_handler()
            logging.basicConfig(filename=os.path.join(self.save_root, "test.log"),
                            format="%(asctime)s %(levelname)s %(message)s", force=True)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            self._logger = logger
        return self._logger

    @property
    def ds_options(self):
        if self._ds_options is None:
            self._ds_cls, self._ds_options = data_provider_solver(self.args, {})
        return self._ds_options

    @property
    def ds_cls(self):
        if self._ds_cls is None:
            __ = self.ds_options
        return self._ds_cls


def ds_from_args(args, rm_keys=True, test_set=False):
    default_kwargs = {'data_root': args["data_root"], 'pre_transform': my_pre_transform, 'record_long_range': True,
                      'type_3_body': 'B', 'cal_3body_term': True}
    if test_set and args["test_set"] is not None:
        dataset_cls, _kwargs = data_provider_solver(args, default_kwargs, ds_key="test_set")
    else:
        dataset_cls, _kwargs = data_provider_solver(args, default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    dataset = dataset_cls(**_kwargs)
    if rm_keys:
        dataset = remove_extra_keys(dataset)
    print("used dataset: {}".format(dataset.processed_file_names))
    if dataset.train_index is not None:
        validate_index(dataset.train_index, dataset.val_index, dataset.test_index)
    if ("add_sol" not in _kwargs or not _kwargs["add_sol"]) and args["data_provider"].split('[')[0] in ["frag9to20_all",
                                                                                                        "frag20_eMol9_combine"]:
        # for some weird logic, I separated training and testing dataset for those two datasets, so I have to deal with
        # it.
        logging.info("swapping {} to frag9to20_jianing".format(args["data_provider"]))
        # no longer supported
        raise NotImplementedError
        frag20dataset, _kwargs = data_provider_solver('frag9to20_jianing', _kwargs)
        _kwargs["training_option"] = "test"
        print(_kwargs)
        return dataset, frag20dataset(**_kwargs)
    else:
        return dataset


def read_folder_config(folder_name):
    # parse config file
    if osp.exists(osp.join(folder_name, "config-test.txt")):
        config_name = osp.join(folder_name, "config-test.txt")
    else:
        config_name = glob.glob(osp.join(folder_name, 'config-*.txt'))[0]
    args = read_config_file(config_name)
    return args, config_name

def read_config_file(config_file):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    args, unknown = parser.parse_known_args(["@" + config_file])
    args = vars(args)
    args = preprocess_config(args)
    return args