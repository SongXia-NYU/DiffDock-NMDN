import copy
from functools import cached_property
import glob
import json
import logging
import math
import os
import os.path as osp
import shutil
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import tqdm
from torch.utils.data import Subset
from torch_geometric.data import DataLoader

from utils.configs import Config, read_config_file, read_folder_config
from utils.eval.evaluator import Evaluator
from utils.train.trainer import data_provider_solver
from utils.DataPrepareUtils import my_pre_transform
from utils.data.DummyIMDataset import DummyIMDataset
from utils.data.LargeDataset import LargeDataset
from utils.eval.trained_folder import TrainedFolder, ds_from_args
from utils.utils_functions import lazy_property, remove_handler, init_model_test, get_device


class Tester(TrainedFolder):
    """
    Load a trained folder for performance evaluation
    """

    def __init__(self, folder_name, use_exist=False,
                 ignore_train=True, config_folder=None, no_runtime_split=False,
                 explicit_ds_config=None, only_predict=False, use_tqdm=False, overwrite_args=None, **kwargs):
        super().__init__(folder_name, config_folder)
        self.overwrite_args = overwrite_args
        self.use_tqdm = use_tqdm
        self.only_predict = only_predict
        self.explicit_ds_config = explicit_ds_config
        self.ignore_train = ignore_train
        self.no_runtime_split = no_runtime_split
        self.use_exist = use_exist
        # these will be used to overwrite self.args for additional functionalities
        self.additional_args: dict = kwargs
        
        self._explicit_ds_args = None
        self._loss_fn = None

        for key in kwargs:
            print("Unused kwargs: ", key)

    @cached_property
    def model(self):
        model_path = osp.join(self.folder_name, 'best_model.pt')
        model_data = torch.load(model_path, map_location=get_device())
        if self.overwrite_args is not None:
            # temp fix
            for key in list(model_data.keys()):
                if key.endswith("_BN"):
                    model_data[key.split("_BN")[0]+"_LIGAND"] = model_data.pop(key)
        # ds was manually set to None
        net = init_model_test(self.explicit_ds_args if self.explicit_ds_args else self.cfg, 
                                model_data, ds=None)
        return net

    @cached_property
    def cfg(self) -> Config:
        cfg: Config = super().cfg
        if "no_pkd_score" in self.additional_args:
            cfg.training.loss_fn.no_pkd_score = self.additional_args["no_pkd_score"]
        if "diffdock_nmdn_result" in self.additional_args:
            cfg.data.diffdock_nmdn_result = self.additional_args["diffdock_nmdn_result"]
        if "linf9_csv" in self.additional_args:
            cfg.data.pre_computed.linf9_csv = self.additional_args["linf9_csv"]
        return cfg

    @cached_property
    def loss_fn(self):
        loss_fn = super(Tester, self).loss_fn
        if self.only_predict: loss_fn.inference_mode()
        return loss_fn

    @property
    def ds(self):
        if self._data_provider is None:
            if self.explicit_ds_config is None:
                return super(Tester, self).ds
            
            _data_provider = None
            default_kwargs = {'data_root': self.explicit_ds_args.data.data_root, 'pre_transform': my_pre_transform,
                    'record_long_range': True, 'type_3_body': 'B', 'cal_3body_term': True}
            ds_cls, ds_args = data_provider_solver(self.explicit_ds_args, default_kwargs)
            # if issubclass(ds_cls, (DummyIMDataset, ESMGearNetProtLig)):
            if issubclass(ds_cls, (DummyIMDataset, )):
                ds_files = glob.glob(osp.join(ds_args["data_root"], "processed", ds_args["dataset_name"]))
                assert len(ds_files) > 0
                if len(ds_files) > 1:
                    # multiple dataset are tested. currently only used for CASF-2016 screening
                    _data_provider = []
                    for ds_file in ds_files:
                        ds_args["dataset_name"] = osp.relpath(ds_file, osp.join(ds_args["data_root"], "processed"))
                        ds_args_cp = copy.deepcopy(ds_args)
                        _data_provider.append((ds_cls, ds_args_cp))
                    _data_provider.sort(key=lambda t: t[1]["dataset_name"])

            if _data_provider is None:
                _data_provider = ds_from_args(self.explicit_ds_args, rm_keys=False)
            self._data_provider = _data_provider
            self._data_provider_test = _data_provider
        return super(Tester, self).ds

    @property
    def explicit_ds_args(self) -> Config:
        if self.explicit_ds_config is None:
            return None

        if self._explicit_ds_args is not None:
            return self._explicit_ds_args
        
        explicit_cfg = OmegaConf.load(self.explicit_ds_config)
        explicit_cfg = OmegaConf.merge(self.cfg, explicit_cfg)
        self._explicit_ds_args = explicit_cfg
        return self._explicit_ds_args

    @property
    def save_root(self) -> str:
        if self._test_dir is None:
            if self.explicit_ds_config is not None:
                test_prefix = self.cfg.folder_prefix + f"_test_on_{self.explicit_ds_args.short_name}_"
            else:
                test_prefix = self.cfg.folder_prefix + '_test_'
            if self.use_exist:
                exist_folders = glob.glob(osp.join(self.folder_name, osp.basename(test_prefix)+"*"))
                if len(exist_folders) > 0:
                    self._test_dir = exist_folders[-1]
                    return self._test_dir
            current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            tmp = test_prefix + current_time
            self._test_dir = osp.join(self.folder_name, osp.basename(tmp))
            os.mkdir(self._test_dir)
        return self._test_dir

    def run_test(self):
        shutil.copy(self.config_name, self.save_root)
        self.info("dataset in args: {}".format(self.cfg.data.data_provider))

        # ---------------------------------- Testing -------------------------------- #

        if not self.no_runtime_split and osp.exists(osp.join(self.folder_name, "runtime_split.pt")):
            explicit_split = torch.load(osp.join(self.folder_name, "runtime_split.pt"))
            # now I suffer the result from the "crime" of mixing "valid" with "val"
            explicit_split["val_index"] = explicit_split["valid_index"]
        else:
            explicit_split = None

        if isinstance(self.ds_test, list):
            for ds_cls, ds_args in self.ds_test:
                this_ds = ds_cls(**ds_args)
                print(f"testing on {this_ds.processed_file_names}")
                self.info("dataset: {}".format(this_ds.processed_file_names))
                self.record_name(this_ds)
                self.eval_ds(this_ds, explicit_split, "test_index")
        else:
            self.record_name(self.ds_test)

            self.info("dataset: {}".format(self.ds.processed_file_names))
            self.info("dataset test: {}".format(self.ds_test.processed_file_names))
            for index_name in ["train_index", "val_index", "test_index"]:
                if index_name == "test_index":
                    this_ds = self.ds_test
                else:
                    this_ds = self.ds
                self.eval_ds(this_ds, explicit_split, index_name)

        self.save_chk()
        # remove global variables
        remove_handler()

    def record_name(self, this_ds):
        # record extra information from the dataset used for docking score and screening score calculation
        # do nothing if self.explicit_ds_config is not specified
        if self.explicit_ds_config is None:
            return
        
        def _conv_d(this_info):
            if isinstance(this_info, torch.Tensor):
                this_info = this_info.cpu().numpy().tolist()
            elif isinstance(this_info, list):
                if isinstance(this_info[0], list):
                    this_info = [item[0] for item in this_info]
            return this_info
        ds_name = osp.basename(this_ds.processed_file_names[0]).split(".pyg")[0]
        if self.explicit_ds_args["record_name"] is not None:
            info = {}
            # if isinstance(this_ds, ESMGearNetProtLig):
            #     this_ds = this_ds.lig_ds

            if isinstance(this_ds, DummyIMDataset):
                for key in self.explicit_ds_args["record_name"]:
                    this_info = getattr(this_ds.data, key)
                    info[key] = _conv_d(this_info)
            else:
                assert isinstance(this_ds, LargeDataset)
                tensor_keys = set()
                for i in range(len(this_ds)):
                    this_d = this_ds[i]
                    for key in self.explicit_ds_args["record_name"]:
                        this_info = getattr(this_d, key)
                        if key not in info:
                            info[key] = []
                            if isinstance(this_info, torch.Tensor):
                                tensor_keys.add(key)
                        info[key].append(_conv_d(this_info))
                for key in tensor_keys:
                    b4concat = [d.view(1, -1) for d in info[key]]
                    info[key] = torch.concat(b4concat, dim=0)
            
            info_df = pd.DataFrame(info)
            info_df.to_csv(osp.join(self.save_root, f"record_name_{ds_name}.csv"), index=False)

    @lazy_property
    def evaluator(self) -> Evaluator:
        return Evaluator(False, self.explicit_ds_args)

    def eval_ds(self, this_ds, explicit_split, index_name):
        self.info(f"Testing on {index_name}")
        if self.ignore_train and index_name == "train_index":
            return
        
        index_short = index_name.split("_")[0]
        ds_name = osp.basename(this_ds.processed_file_names[0]).split(".pyg")[0]
        result_name = f"loss_{ds_name}_{index_short}.pt"
        result_file = osp.join(self.save_root, result_name)
        if osp.exists(result_file):
            print(f"{osp.abspath(result_file)} exists, skipping...")
            self.info(f"{osp.abspath(result_file)} exists, skipping...")
            return

        if explicit_split is not None:
            this_index = explicit_split[index_name]
        else:
            this_index = getattr(this_ds, index_name)
        if this_index is None:
            # for external test datasets where train_index and val_index are None
            return
        
        this_index = torch.as_tensor(this_index)

        self.info("{} size: {}".format(index_name, len(this_index)))

        batch_size = self.explicit_ds_args.training.valid_batch_size
        test_ds: Subset = Subset(this_ds, torch.as_tensor(this_index).tolist())
        this_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        if self.use_tqdm:
            this_dl = tqdm.tqdm(this_dl, desc=index_name, total=len(this_ds)//batch_size)

        result = self.evaluator(self.model, this_dl, self.loss_fn)
        result["target_names"] = self.loss_fn.target_names
        torch.save(result, result_file)

        self.info("-------------- {} ---------------".format(index_short))
        for key in result:
            d = result[key]
            if isinstance(d, torch.Tensor):
                if torch.numel(d) == 1:
                    self.info("{}: {}".format(key, result[key].item()))
                else:
                    self.info("{} with shape: {}".format(key, result[key].shape))
            else:
                self.info("{}: {}".format(key, result[key]))
        self.info("----------- end of {} ------------".format(index_short))

    def save_chk(self):
        chk_dict = {}
        vars_self = vars(self)
        for key in vars_self.keys():
            if isinstance(vars_self[key], (int, float, str, bool)):
                chk_dict[key] = vars_self[key]
        with open(osp.join(self.save_root, "chk.json"), "w") as f:
            json.dump(chk_dict, f, indent=2)

    def load_chk(self):
        with open(osp.join(self.save_root, "chk.json")) as f:
            chk_dict = json.load(f)
        for key in chk_dict.keys():
            # backward compatibility
            if not hasattr(self, key):
                continue
            if getattr(self, key) is None:
                setattr(self, key, chk_dict[key])


def print_uncertainty_figs(pred_std, diff, name, unit, test_dir, n_bins=10):
    import matplotlib.pyplot as plt
    # let x-axis ordered ascending
    std_rank = torch.argsort(pred_std)
    pred_std = pred_std[std_rank]
    diff = diff[std_rank]

    diff = diff.abs()
    diff_2 = diff ** 2
    x_data = torch.arange(pred_std.min(), pred_std.max(), (pred_std.max() - pred_std.min()) / n_bins)
    mae_data = torch.zeros(x_data.shape[0] - 1).float()
    rmse_data = torch.zeros_like(mae_data)
    for i in range(x_data.shape[0] - 1):
        mask = (pred_std < x_data[i + 1]) & (pred_std > x_data[i])
        mae_data[i] = diff[mask].mean()
        rmse_data[i] = torch.sqrt(diff_2[mask].mean())

    plt.figure(figsize=(15, 10))
    # Plotting predicted error MAE vs uncertainty
    plt.plot(x_data[1:], mae_data, label='{} MAE, {}'.format(name, unit))
    plt.plot(x_data[1:], rmse_data, label='{} RMSE, {}'.format(name, unit))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.savefig(os.path.join(test_dir, 'avg_error_uncertainty'))

    fig, ax1 = plt.subplots(figsize=(15, 10))
    diff_abs = diff.abs()
    ax1.scatter(pred_std, diff_abs, alpha=0.1)
    ax1.set_xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax1.set_ylabel('Error of {}, {}'.format(name, unit))
    # plt.title('Uncertainty vs. prediction error')
    # plt.savefig(os.path.join(test_dir, 'uncertainty'))

    # Plotting cumulative large error percent vs uncertainty
    thresholds = ['0', '1.0', '10.0']
    cum_large_count = {threshold: torch.zeros_like(x_data) for threshold in thresholds}
    for i in range(x_data.shape[0]):
        mask = (pred_std < x_data[i])
        select_diff = diff[mask]
        for threshold in thresholds:
            cum_large_count[threshold][i] = select_diff[select_diff > float(threshold)].shape[0]
    # plt.figure(figsize=(15, 10))
    ax2 = ax1.twinx()
    for threshold in thresholds:
        count = cum_large_count[threshold]
        ax2.plot(x_data, count / count[-1] * 100,
                 label='%all molecules' if threshold == '0' else '%large>{}kcal/mol'.format(threshold))

    plt.legend()
    # ax2.xlabel('Uncertainty of {}, {}'.format(name, unit))
    ax2.set_ylabel('percent of molecules')
    plt.savefig(os.path.join(test_dir, 'percent'))

    # Plotting density of large error percent vs uncertainty
    x_mid = (x_data[:-1] + x_data[1:]) / 2
    plt.figure(figsize=(15, 10))
    for i, threshold in enumerate(thresholds):
        count_density_all = cum_large_count['0'][1:] - cum_large_count['0'][:-1]
        count_density = cum_large_count[threshold][1:] - cum_large_count[threshold][:-1]
        count_density_lower = count_density_all - count_density
        width = (x_data[1] - x_data[0]) / (len(thresholds) * 5)
        plt.bar(x_mid + i * width, count_density / count_density.sum() * 100, width=width,
                label='all molecules' if threshold == '0' else 'large>{}kcal/mol'.format(threshold))
        # if threshold != '0':
        #     plt.bar(x_mid - i * width, count_density_lower / count_density_lower.sum() * 100, width=width,
        #             label='large<{}kcal/mol'.format(threshold))
    plt.legend()
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('density of molecules')
    plt.xticks(x_data)
    plt.savefig(os.path.join(test_dir, 'percent_density'))

    # box plot
    num_points = diff.shape[0]
    step = math.ceil(num_points / n_bins)
    sep_index = torch.arange(0, num_points + 1, step)
    y_blocks = []
    x_mean = []
    for num, start in enumerate(sep_index):
        if num + 1 < sep_index.shape[0]:
            end = sep_index[num + 1]
        else:
            end = num_points
        y_blocks.append(diff[start: end].numpy())
        x_mean.append(pred_std[start: end].mean().item())
    plt.figure(figsize=(15, 10))
    box_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.boxplot(y_blocks, notch=True, positions=x_mean, vert=True, showfliers=False,
                widths=box_size)
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('Error of {}, {}'.format(name, unit))
    plt.title('Boxplot')
    plt.savefig(os.path.join(test_dir, 'box_plot'))

    # interval percent
    small_err_percent_1 = np.asarray([(x < 1.).sum() / x.shape[0] for x in y_blocks])
    large_err_percent_10 = np.asarray([(x > 10.).sum() / x.shape[0] for x in y_blocks])
    x_mean = np.asarray(x_mean)
    plt.figure(figsize=(15, 10))
    bar_size = (0.30 * (x_mean[-1] - x_mean[0]) / len(x_mean))
    plt.bar(x_mean - bar_size, large_err_percent_10, label='percent, error > 10kcal/mol', width=bar_size)
    plt.bar(x_mean, 1 - large_err_percent_10 - small_err_percent_1,
            label='percent, 1kcal/mol < error < 10kcal/mol', width=bar_size)
    plt.bar(x_mean + bar_size, small_err_percent_1, label='small error < 1kcal/mol', width=bar_size)
    plt.legend()
    plt.xticks(x_mean, ['{:.3f}'.format(_x_mean) for _x_mean in x_mean])
    plt.xlim([x_mean[0] - box_size, x_mean[-1] + box_size])
    plt.xlabel('Uncertainty of {}, {}'.format(name, unit))
    plt.ylabel('percent')
    plt.title('error percent')
    plt.savefig(os.path.join(test_dir, 'error_percent'))
    return


def test_info_analyze(pred, target, test_dir, logger=None, name='Energy', threshold_base=1.0, unit='kcal/mol',
                      pred_std=None, x_forward=0):
    diff = pred - target
    rank = torch.argsort(diff.abs())
    diff_ranked = diff[rank]
    if logger is None:
        logging.basicConfig(filename=os.path.join(test_dir, 'test.log'),
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        remove_logger = True
    else:
        remove_logger = False
    logger.info('Top 10 {} error: {}'.format(name, diff_ranked[-10:]))
    logger.info('Top 10 {} error id: {}'.format(name, rank[-10:]))
    e_mae = diff.abs().mean()
    logger.info('{} MAE: {}'.format(name, e_mae))
    thresholds = torch.logspace(-2, 2, 50) * threshold_base
    thresholds = thresholds.tolist()
    thresholds.extend([1.0, 10.])
    for threshold in thresholds:
        mask = (diff.abs() < threshold)
        logger.info('Percent of {} error < {:.4f} {}: {} out of {}, {:.2f}%'.format(
            name, threshold, unit, len(diff[mask]), len(diff), 100 * float(len(diff[mask])) / len(diff)))
    torch.save(diff, os.path.join(test_dir, 'mol_lvl_detail.pt'))

    # concrete dropout
    if x_forward and (pred_std is not None):
        # print_scatter(pred_std, mol_lvl_detail, name, unit, test_dir)
        print_uncertainty_figs(pred_std, diff, name, unit, test_dir)
    if x_forward:
        raise NotImplemented
        # print_training_curve(test_dir)

    if remove_logger:
        remove_handler(logger)
    return
