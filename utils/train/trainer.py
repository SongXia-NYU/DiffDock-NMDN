import argparse
import glob
import json
import logging
import math
import os
import os.path as osp
import shutil
import sys
import time
from collections import OrderedDict
from copy import copy
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.cuda
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from tqdm import tqdm

from Networks.PhysDimeNet import PhysDimeNet
from Networks.UncertaintyLayers.swag import SWAG
from utils.DataPrepareUtils import my_pre_transform, rm_atom
from utils.data.DummyIMDataset import AuxPropDataset, DummyIMDataset, VSDummyIMDataset, VSPointerDummyIMDataset
from utils.data.data_utils import data_to_device, get_lig_natom
from utils.data.delta_learning_ds import CASFSoringDeltaLearningDS, CASFDockingDeltaLearningDS, CASFScreeningDeltaLearningDS, PLDeltaLearningDataset
from utils.data.LargeDataset import LargeDataset
from utils.LossFn import BaseLossFn
from utils.Optimizers import EmaAmsGrad, MySGD
from utils.data.prot_embedding_ds import PPEmbedDS, ProteinEmbeddingDS
from utils.data.vs_im_datasets import ChunkMapperDataset
from utils.eval.evaluator import Evaluator
from utils.tags import tags
from utils.time_meta import print_function_runtime, record_data
from utils.utils_functions import add_parser_arguments, lossfn_factory, floating_type, lazy_property, mp_mean_std_calculate, preprocess_config, get_lr, get_n_params, \
    atom_mean_std, remove_handler, option_solver, fix_model_keys, get_device, process_state_dict, \
    validate_index, non_collapsing_folder, solv_num_workers

from ocpmodels.models.equiformer_v2.edge_rot_mat import InitEdgeRotError


class Trainer:
    def __init__(self, config_args, data_provider=None):
        self.config_args = config_args

        self._config_processed = None
        self._time_it = None
        self._run_directory = None
        self._sum_writer = None
        self._is_main = None
        self._logger = None
        self._meta_data_path = None
        self._loss_csv = None
        self._loss_df = None

        self._train_index = None
        self._train_size = None
        self._val_index = None
        self._test_index = None
        self._train_loader = None
        self._val_loader = None

        self._ds = data_provider
        self._ds_args = None

        # initialized during training
        self.train_model = None
        self.scheduler = None
        self.optimizer = None
        self.early_stop_count: int = 0

    def train(self, explicit_split=None, ignore_valid=False, use_tqdm=False):
        if explicit_split is not None:
            if is_main:
                torch.save(explicit_split, osp.join(self.run_directory, "runtime_split.pt"))
            if isinstance(explicit_split, tuple):
                self._train_index, self._val_index, self._test_index = explicit_split
            elif isinstance(explicit_split, dict):
                self._train_index, self._val_index, self._test_index = \
                    explicit_split["train_index"], explicit_split["valid_index"], explicit_split["test_index"]
            else:
                raise ValueError(f"cannot understand split: {explicit_split}")
        
        # ------------------- variable set up ---------------------- #
        config_dict = self.config_processed

        # record run time of programs
        if self.time_it:
            assert not config_dict["is_dist"]
        self.t0 = time.time()

        local_rank = config_dict["local_rank"]
        is_main = self.is_main
        if torch.cuda.is_available() and config_dict["is_dist"]:
            torch.cuda.set_device(local_rank)
            # ---------------- Distributed Training ------------------- #
            self.log(f"start init..., gpu = {local_rank}")
            dist.init_process_group(backend='nccl', init_method='env://')
            self.log(f"end init..., gpu = {local_rank}")

        self.log("used dataset: {}".format(self.ds.processed_file_names))

        loss_fn = self.loss_fn

        # ------------------- Setting up model and optimizer ------------------ #
        mean_atom, std_atom = self.get_norms()

        config_dict['energy_shift'] = mean_atom
        config_dict['energy_scale'] = std_atom

        if config_dict["ext_atom_features"] is not None:
            # infer the dimension of external atom feature
            ext_atom_feature = getattr(self.ds[[0]].data, config_dict["ext_atom_features"])
            ext_atom_dim = ext_atom_feature.shape[-1]
            config_dict["ext_atom_dim"] = ext_atom_dim
            del ext_atom_feature

        train_model = PhysDimeNet(cfg=config_dict, ds=self.ds, **config_dict)
        train_model = train_model.to(get_device())
        # train_model = train_model.type(floating_type)

        if config_dict["use_swag"]:
            dummy_model = PhysDimeNet(cfg=config_dict, **config_dict).to(get_device()).type(floating_type)
            swag_model = SWAG(dummy_model, no_cov_mat=False, max_num_models=20)
        else:
            swag_model = None

        # model freeze options (transfer learning)
        if config_dict["freeze_option"] == 'prev':
            train_model.freeze_prev_layers(freeze_extra=False)
        elif config_dict["freeze_option"] == 'prev_extra':
            train_model.freeze_prev_layers(freeze_extra=True)
        elif config_dict["freeze_option"] == 'none':
            pass
        else:
            raise ValueError('Invalid freeze option: {}'.format(config_dict["freeze_option"]))

        # ---------------------- restore pretrained model ------------------------ #
        def get_pretrained_paths(cfg: dict) -> Tuple[str, str, Optional[str]]:
            model_chk = cfg["chk"] if cfg["chk"] else cfg["use_trained_model"]
            if not osp.isdir(model_chk):
                # the pth file is directly provided.
                return model_chk, model_chk, None

            # the pth file is infered from the directory
            trained_model_dir = glob.glob(model_chk)
            assert len(trained_model_dir) == 1, f"Zero or multiple trained folder: {trained_model_dir}"
            trained_model_dir = trained_model_dir[0]
            cfg["use_trained_model"] = trained_model_dir
            if is_main:
                self.log('using trained model: {}'.format(trained_model_dir))
            train_model_path = osp.join(trained_model_dir, 'training_model.pt')
            if cfg["ft_discard_training_model"] or not osp.exists(train_model_path):
                train_model_path = osp.join(trained_model_dir, 'best_model.pt')
            val_model_path = osp.join(trained_model_dir, 'best_model.pt')
            return train_model_path, val_model_path, trained_model_dir
        
        if config_dict["use_trained_model"] or config_dict["chk"]:
            train_model_path, val_model_path, trained_model_dir = get_pretrained_paths(config_dict)
            train_dict = torch.load(train_model_path, map_location=get_device())
            train_dict = fix_model_keys(train_dict)
            train_dict = process_state_dict(train_dict, config_dict, self.logger, is_main)

            incompatible_keys = train_model.load_state_dict(state_dict=train_dict, strict=False)
            if is_main:
                self.log(f"---------vvvvv incompatible keys in {train_model_path} vvvvv---------")
                for name in ["missing_keys", "unexpected_keys"]:
                    self.log(name.upper()+" :")
                    for key in getattr(incompatible_keys, name):
                        self.log(key)
                    self.log("*"*10)

            shadow_dict = torch.load(val_model_path, map_location=get_device())
            shadow_dict = process_state_dict(fix_model_keys(shadow_dict), config_dict, self.logger, is_main)
        else:
            shadow_dict = None
            trained_model_dir = None
        self.train_model = train_model

        # optimizers
        ema_decay = config_dict["ema_decay"]
        if config_dict["optimizer"].split('_')[0] == 'emaAms':
            # for some reason I created two ways of specifying EMA decay...
            # for compatibility, I will keep both
            assert float(config_dict["optimizer"].split('_')[1]) == ema_decay
            optimizer = EmaAmsGrad(train_model, lr=config_dict["learning_rate"],
                                   ema=float(config_dict["optimizer"].split('_')[1]),
                                   shadow_dict=shadow_dict, params=self.get_optim_params(train_model),
                                   use_buffers=config_dict["swa_use_buffers"], run_dir=self.run_directory)
        elif config_dict["optimizer"].split('_')[0] == 'sgd':
            optimizer = MySGD(train_model, lr=config_dict["learning_rate"])
        else:
            raise ValueError('Unrecognized optimizer: {}'.format(config_dict["optimizer"]))
        self.optimizer = optimizer

        # schedulers
        scheduler_kw_args = option_solver(config_dict["scheduler"], type_conversion=True)
        scheduler_base = config_dict["scheduler"].split("[")[0]
        config_dict["scheduler_base"] = scheduler_base
        if scheduler_base == "StepLR":
            if "decay_epochs" in scheduler_kw_args.keys():
                step_per_epoch = 1. * self.train_size / config_dict["batch_size"]
                decay_steps = math.ceil(scheduler_kw_args["decay_epochs"] * step_per_epoch)
            else:
                decay_steps = config_dict["decay_steps"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=0.1)
        elif scheduler_base == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kw_args)
            assert config_dict["warm_up_steps"] == 0
        else:
            raise ValueError('Unrecognized scheduler: {}'.format(config_dict["scheduler"]))

        if config_dict["warm_up_steps"] > 0:
            from warmup_scheduler import GradualWarmupScheduler
            warm_up_scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                                       total_epoch=config_dict["warm_up_steps"],
                                                       after_scheduler=scheduler)
        else:
            warm_up_scheduler = scheduler

        if trained_model_dir and (not config_dict["reset_optimizer"]):
            if os.path.exists(os.path.join(trained_model_dir, "best_model_optimizer.pt")):
                optimizer.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_optimizer.pt"),
                                                     map_location=get_device()))
            if os.path.exists(os.path.join(trained_model_dir, "best_model_scheduler.pt")):
                scheduler.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_scheduler.pt"),
                                                     map_location=get_device()))
        self.scheduler = scheduler

        if config_dict["use_trained_model"]:
            # protect LR been reduced
            step_per_epoch = 1. * self.train_size / config_dict["batch_size"]
            ema_avg = 1 / (1 - ema_decay)
            ft_protection = math.ceil(ema_avg / step_per_epoch)
            ft_protection = min(30, ft_protection)
        else:
            ft_protection = 0
        if is_main:
            self.log(f"Fine tune protection: {ft_protection} epochs. ")

        # --------------------- Printing meta data ---------------------- #
        if get_device().type == 'cuda':
            self.log('Hello from device : ' + torch.cuda.get_device_name(get_device()))
            self.log("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(get_device()) * 1e-6))
        else:
            if not config_dict["debug_mode"]:
                print("You are not using a GPU")
                # assert not osp.exists("/scratch/sx801"), "You are running on HPC but not using a GPU"
                # assert not osp.exists("/archive/sx801"), "You are running on HPC but not using a GPU"

        n_parm, model_structure = get_n_params(train_model, None, False)
        self.log('model params: {}'.format(n_parm))
        self.log_meta('*' * 20 + '\n')
        self.log_meta("all params\n")
        self.log_meta(model_structure)
        self.log_meta('*' * 20 + '\n')
        n_parm, model_structure = get_n_params(train_model, None, True)
        self.log('trainable params: {}'.format(n_parm))
        self.log_meta('*' * 20 + '\n')
        self.log_meta("trainable params\n")
        self.log_meta(model_structure)
        self.log_meta('*' * 20 + '\n')
        self.log_meta('train data index:{} ...\n'.format(self.train_index[:100]))
        self.log_meta('val data index:{} ...\n'.format(self.val_index[:100]))
        # self.log_meta()('test data index:{} ...\n'.format(test_index[:100]))
        for _key in config_dict.keys():
            self.log_meta("{} = {}\n".format(_key, config_dict[_key]))

        # ---------------------- Final steps before training ----------------- #
        if torch.cuda.is_available() and config_dict["is_dist"]:
            # There we go, distributed training!!!
            train_model.cuda(local_rank)
            train_model = DDP(train_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # ---------------------- Training ----------------------- #
        self.log('start training...')

        shadow_net = optimizer.shadow_model
        val_res = val_step_new(shadow_net, self.val_loader, loss_fn)

        valid_info_dict = OrderedDict(
            {"epoch": 0, "train_loss": -1, "valid_loss": val_res["loss"], "delta_time": time.time() - self.t0})
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                valid_info_dict[key] = val_res[key]
                valid_info_dict.move_to_end(key)
        self.update_df(valid_info_dict)

        # use np.inf instead of val_res["loss"] for proper transfer learning behaviour
        self.best_loss = np.inf

        if is_main:
            last_epoch = pd.read_csv(osp.join(self.run_directory, "loss_data.csv"), header="infer").iloc[-1]["epoch"]
            last_epoch = int(last_epoch.item())
            self.log('Init lr: {}'.format(get_lr(optimizer)))
        else:
            last_epoch = 0

        self.step = 0

        if self.time_it:
            self.t0 = record_data("setup", self.t0)

        self.log("Setup complete, training starts...")

        for epoch in range(last_epoch, last_epoch + config_dict["num_epochs"]):
            # let all processes sync up before starting with a new epoch of training
            if torch.cuda.is_available() and config_dict["is_dist"]:
                dist.barrier()

            # Early stop when learning rate is too low
            this_lr = get_lr(optimizer)
            if self.step > config_dict["warm_up_steps"] and config_dict["stop_low_lr"] and \
                this_lr < 3 * getattr(scheduler, "eps", 1e-9):
                self.log('early stop because of low LR at epoch {}.'.format(epoch))
                break

            loader = enumerate(self.train_loader)
            if use_tqdm:
                loader = tqdm(loader, "training", total=len(self.train_loader))

            train_loss = 0.
            for batch_num, data in loader:
                data = data_to_device(data)
                this_size = get_lig_natom(data).shape[0]

                try:
                    train_loss += self.train_step(train_model, _optimizer=optimizer, data_batch=data, loss_fn=loss_fn,
                                                max_norm=config_dict["max_norm"], warm_up_scheduler=warm_up_scheduler,
                                                config_dict=config_dict) * this_size / self.train_size
                except RuntimeError as e:
                    if "CUDA out of memory." in str(e):
                        self.logger.info("| WARNING: ran out of memory, skipping batch")
                        for p in train_model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    elif "CUDA error: device-side assert triggered" in str(e):
                        self.logger.info("CUDA error")
                        self.logger.info(f"{data}")
                        self.logger.info(f"{data.pdb}")
                        self.logger.info("Sad :<")
                        raise e
                    else:
                        raise e
                except InitEdgeRotError as e:
                    self.logger.error(f"init_edge_rot_mat Error: {str(e)}, skipping batch of {data.pdb}.")
                    for p in train_model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                except ValueError as e:
                    if "Expected more than 1 value per channel when training, got input size" in str(e):
                        self.logger.info("Batch normalization error, skipping batch..")
                        for p in train_model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    raise e
                self.step += 1
                if config_dict["eval_per_step"] is not None and (self.step % config_dict["eval_per_step"] == 0):
                    # evaluate every k steps specified in --eval_per_step
                    self.eval_step(train_loss, epoch)

            if config_dict["eval_per_step"] is None:
                # By default, evaluate every epoch
                self.eval_step(train_loss, epoch)
            if config_dict["scheduler_base"] in tags.step_per_epoch and (epoch - last_epoch) >= ft_protection:
                warm_up_scheduler.step(metrics=val_res["loss"])

            if config_dict["use_swag"]:
                start, freq = config_dict["uncertainty_modify"].split('_')[1], \
                                config_dict["uncertainty_modify"].split('_')[2]
                if epoch > int(start) and (epoch % int(freq) == 0):
                    swag_model.collect_model(shadow_net)
                    torch.save(swag_model.state_dict(), osp.join(self.run_directory, 'swag_model.pt'))

        if is_main:
            if self.time_it:
                self.t0 = record_data("training", self.t0)
                print_function_runtime(folder=self.run_directory)
            remove_handler(self.logger)
        meta = {"run_directory": self.run_directory}
        return meta

    def eval_step(self, train_loss, epoch: int):
        # ---------------------- Evaluation step: validation, save model, print meta ---------------- #
        if not self.is_main:
            return
        this_lr = get_lr(self.optimizer)
        loss_fn = self.loss_fn
        cfg = self.config_processed

        self.log('step {} ended, learning rate: {} '.format(self.step, this_lr))
        shadow_net = self.optimizer.shadow_model
        # self.log("prepare validation")
        val_res = val_step_new(shadow_net, self.val_loader, loss_fn)
        # self.log("scheduler stepping")

        # _loss_data_this_epoch = {'epoch': epoch, 't_loss': train_loss, 'v_loss': val_res['loss'],
        #                          'time': time.time()}
        # _loss_data_this_epoch.update(val_res)
        # loss_data.append(_loss_data_this_epoch)
        # torch.save(loss_data, os.path.join(run_directory, 'loss_data.pt'))

        # self.log("prepare log dict")
        valid_info_dict = {"step": self.step, "epoch": epoch, "train_loss": train_loss, 
                           "valid_loss": val_res["loss"], "delta_time": time.time() - self.t0}
        # self.log("updating log dict")
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                valid_info_dict[key] = val_res[key]
        # self.log("updating df")
        self.update_df(valid_info_dict)
        self.summarize_epoch(**valid_info_dict)
        self.t0 = time.time()

        if val_res['loss'] < self.best_loss:
            self.early_stop_count = 0
            self.best_loss = val_res['loss']
            torch.save(shadow_net.state_dict(), osp.join(self.run_directory, 'best_model.pt'))
            torch.save(self.train_model.state_dict(), osp.join(self.run_directory, 'training_model.pt'))
            torch.save(self.optimizer.state_dict(), osp.join(self.run_directory, 'best_model_optimizer.pt'))
            torch.save(self.scheduler.state_dict(), osp.join(self.run_directory, "best_model_scheduler.pt"))
        else:
            self.early_stop_count += 1
            if self.early_stop_count == cfg["early_stop"]:
                self.log('early stop at epoch {}.'.format(epoch))
                return True
        return False

    def get_norms(self):
        if not self.config_processed["normalize"]:
            return 0., 1.
        train_index = self.train_index
        # skip scale and shift calculation if transfer learning
        if self.config_processed["use_trained_model"] and not self.config_processed["reset_scale_shift"]:
            return 0. ,1.
        
        # scale and shift not used in MDN loss
        if self.config_processed["loss_metric"] == "mdn":
            return 0. ,1.

        # Normalization of PhysNet atom-wise prediction
        if isinstance(self.ds, LargeDataset):
            # Those are pre-calculated to reduce CPU time
            if osp.basename(self.ds_args["file_locator"]) == "PDBind_v2020_10$10$6.loc.pth":
                return 0.0180, 0.0121
            elif osp.basename(self.ds_args["file_locator"]).startswith("PL_train"):
                return 0.0177, 0.0094
            elif osp.basename(self.ds_args["split"]) == "PL2020-polarH_10$10$6.split_all.pth":
                return 0.0107, 0.0047
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt500-MartiniMD.thres5000.split.pth":
                return 4.6173, 6.0784
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt500-MartiniOPT.thres0.split.pth":
                return -13.7419, 1.8257
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt500-GBSAWaterOPT-Martini.thres0.split.pth":
                return -16.6572, 3.0555
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt-Martini-c10.thres0.split.pth":
                return torch.as_tensor([-7.2597, -14.7623, -14.1213], dtype=floating_type), \
                        torch.as_tensor([2.7736, 2.7279, 2.6224], dtype=floating_type)
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt-Martini-c10.thres99999.split.pth":
                return torch.as_tensor([-6.4441, -14.7366, -14.0281], dtype=floating_type), \
                        torch.as_tensor([4.3027, 2.9967, 2.8648], dtype=floating_type)
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt-Martini-EGB-c10.split.pth":
                return torch.as_tensor([0.0000, -8.2594, -7.5537], dtype=floating_type), \
                        torch.as_tensor([0.0000, 4.3943, 4.0188], dtype=floating_type)
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt-FRAG1To5-GBSA-c10.split.pth":
                return torch.as_tensor([-2.3097, -33.9814, -31.2755], dtype=floating_type), \
                        torch.as_tensor([12.8616, 16.2893, 15.4502], dtype=floating_type)
            elif osp.basename(self.ds_args["split"]) in ["AF-SwissProt-CAP-FRAG1To2-Martini-GBSA-c10.split.pth"]:
                return torch.as_tensor([-7.4477, -23.2636, -21.9126], dtype=floating_type), \
                        torch.as_tensor([12.1725, 15.5938, 15.0134], dtype=floating_type)
            elif osp.basename(self.ds_args["split"]) == "AF-SwissProt-CAP-FRAG1To2-CapMartini-GBSA-c10.split.pth":
                return torch.as_tensor([-4.6798, -14.6433, -13.7921], dtype=floating_type), \
                        torch.as_tensor([7.8141, 10.0154,  9.6566], dtype=floating_type)
            else:
                print("WARNING!!!!!!! Please save those pre-calculated value to reduce CPU time.")
                self.log("WARNING!!!!!!! Please save those pre-calculated value to reduce CPU time.")
                mean_atom, std_atom = mp_mean_std_calculate(self.ds, train_index, self.config_processed, self.run_directory)
        elif osp.basename(self.ds_args["dataset_name"]).startswith("PL_train-"):
            # those numbers are calculated based on pKd and number of ligand and surrounding protein atoms
            return 0.0177, 0.0094
        elif self.config_processed["action"] == "E":
            mean_atom, std_atom = atom_mean_std(getattr(self.ds.data, self.config_processed["action"]),
                                                self.ds.data.N, train_index)
            mean_atom = mean_atom.item()
        elif self.config_processed["action"] in ["names", "names_and_QD"]:
            mean_atom = []
            std_atom = []
            for name in self.config_processed["target_names"]:
                N_matrix = self.ds.data.N_l if self.config_processed["ligand_only"] else self.ds.data.N
                this_mean, this_std = atom_mean_std(getattr(self.ds.data, name), N_matrix, train_index)
                mean_atom.append(this_mean)
                std_atom.append(this_std)
            if self.config_processed["action"] == "names_and_QD":
                # the last dimension is for predicting atomic charge
                mean_atom.append(0.)
                std_atom.append(1.)
            mean_atom = torch.as_tensor(mean_atom)
            std_atom = torch.as_tensor(std_atom)
        elif self.config_processed["action"] == "names_atomic":
            mean_atom = []
            std_atom = []
            for name in self.config_processed["target_names"]:
                this_atom_prop: torch.Tensor = getattr(self.ds.data, name)
                mean_atom.append(torch.mean(this_atom_prop.type(floating_type)).item())
                std_atom.append(torch.std(this_atom_prop.type(floating_type)).item())
            mean_atom = torch.as_tensor(mean_atom)
            std_atom = torch.as_tensor(std_atom)
        else:
            raise ValueError("Invalid action: {}".format(self.config_processed["action"]))

        return mean_atom, std_atom

    def train_step(self, model, _optimizer, data_batch, loss_fn, max_norm, warm_up_scheduler, config_dict):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # self.t0 = time.time()
        with torch.autograd.set_detect_anomaly(False):
            model.train()
            if config_dict["mdn_freeze_bn"]:
                # turn off the batch normalization in the MDN layer for proper fine-tune behaviour
                model.mdn_layer.MLP[1].eval()
            _optimizer.zero_grad()

            model_out = model(data_batch)

            if self.time_it:
                self.t0 = record_data('forward', self.t0, True)

            loss = loss_fn(model_out, data_batch, True) + model_out["nh_loss"]

            if self.time_it:
                self.t0 = record_data('loss_cal', self.t0, True)

            loss.backward()

            if self.time_it:
                self.t0 = record_data('backward', self.t0, True)

        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=config_dict["error_if_nonfinite"])
        except RuntimeError as e:
            if "is non-finite, so it cannot be clipped" not in str(e):
                raise e
            # the NaN gradient error happens when training the cross_mdn_pkd loss
            logging.warning("SKIPPING batch: " + str(e))
            return 0.
        _optimizer.step()
        if config_dict["scheduler_base"] in tags.step_per_step:
            warm_up_scheduler.step()

        if self.time_it:
            self.t0 = record_data('step', self.t0, True)
        # print_function_runtime()

        result_loss = loss.data[0].cpu().item()

        return result_loss

    def get_optim_params(self, model):
        if self.config_processed["ft_lr_factor"] is None:
            return model.parameters()

        normal_lr_parms = []
        reduced_lr_parms = []
        reduced_lr = self.config_processed["learning_rate"]*self.config_processed["ft_lr_factor"]
        freeze: bool = (self.config_processed["ft_lr_factor"] == 0)
        normal_lr_ptns = [re.compile(ptn) for ptn in self.config_processed["normal_lr_ptn"]] if self.config_processed["normal_lr_ptn"] else []
        lower_lr_ptns = [re.compile(ptn) for ptn in self.config_processed["lower_lr_ptn"]] if self.config_processed["lower_lr_ptn"] else []
        # exclusive
        if len(normal_lr_ptns) > 0: assert len(lower_lr_ptns) == 0
        if len(lower_lr_ptns) > 0: assert len(normal_lr_ptns) == 0
        def decide_normal_lr(param_name: str) -> bool:
            for ptn in normal_lr_ptns:
                if ptn.fullmatch(param_name): return True
            if len(normal_lr_ptns) > 0: return False

            assert len(lower_lr_ptns) > 0
            for ptn in lower_lr_ptns:
                if ptn.fullmatch(param_name): return False
            return True
        
        self.log(f"These params have reduced LR of {reduced_lr}: >>>>>>>>>>>>>>")
        if freeze:
            self.log("Well they will actually be freezed...")
        for param_name, param in model.named_parameters():
            normal_lr = decide_normal_lr(param_name)
            if normal_lr:
                normal_lr_parms.append(param)
                continue
            # reduced learning rate parameters
            self.log(param_name)
            reduced_lr_parms.append(param)
            if freeze:
                param.requires_grad = False
        self.log("<<<<<<<<<<<<<<<<<<<<")
        return [{"params": normal_lr_parms},
                {"params": reduced_lr_parms,
                 "lr": reduced_lr}]

    def update_df(self, info_dict: dict):
        if not self.is_main:
            return

        new_df = pd.DataFrame(info_dict, index=[0])
        updated = pd.concat([self.loss_df, new_df])
        updated.to_csv(self.loss_csv, index=False)
        self.set_loss_df(updated)

    def log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_meta(self, msg):
        if not self.is_main:
            return

        with open(self.meta_data_path, "a") as f:
            f.write(msg)
        return

    def set_loss_df(self, df: pd.DataFrame):
        if self._loss_df is None:
            # init
            __ = self.loss_df
        self._loss_df = df

    def summarize_epoch(self, step, delta_time, **scalars):
        for key in scalars:
            self.sum_writer.add_scalar(key, scalars[key], global_step=step, walltime=delta_time)

    @property
    def ds(self):
        if self._ds is None:
            self._ds, self._ds_args = dataset_from_args(self.config_processed, self.logger, return_ds_args=True)
        return self._ds

    @property
    def ds_args(self):
        if self._ds_args is None:
            __ = self.ds
        return self._ds_args

    @property
    def sum_writer(self):
        if self._sum_writer is None:
            self._sum_writer = SummaryWriter(self.run_directory)
        return self._sum_writer

    @property
    def loss_df(self):
        if self._loss_df is None:
            if osp.exists(self.loss_csv):
                # load pretrained folder csv
                loss_df = pd.read_csv(self.loss_csv)
            else:
                loss_df = pd.DataFrame()
            self._loss_df = loss_df
        return self._loss_df

    @property
    def loss_csv(self):
        if self._loss_csv is None:
            self._loss_csv = osp.join(self.run_directory, "loss_data.csv")
        return self._loss_csv

    @property
    def meta_data_path(self):
        if self._meta_data_path is None:
            self._meta_data_path = os.path.join(self.run_directory, 'meta.txt')
        return self._meta_data_path

    @property
    def logger(self):
        if not self.is_main:
            return None

        if self._logger is None:
            # --------------------- Logger setup ---------------------------- #
            # first we want to remove previous logger step up by other programs
            # we sometimes encounter issues and the logger file doesn't show up
            log_tmp = logging.getLogger()
            remove_handler(log_tmp)
            logging.basicConfig(filename=osp.join(self.run_directory, self.config_processed["log_file_name"]),
                                format='%(asctime)s %(message)s', filemode='w')
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            self._logger = logger
        return self._logger

    @property
    def is_main(self):
        if self._is_main is None:
            local_rank = self.config_processed["local_rank"]
            is_main: bool = (local_rank == 0)
            self._is_main = is_main
        return self._is_main

    @property
    def run_directory(self):
        if not self.is_main:
            return None

        if self._run_directory is None:
            # ----------------- set up run directory -------------------- #
            if self.config_processed["chk"] is None:
                folder_prefix = self.config_processed["folder_prefix"]
                tmp = osp.basename(folder_prefix)
                run_directory = non_collapsing_folder(folder_prefix)
                shutil.copyfile(self.config_processed["config_name"], osp.join(run_directory, f"config-{tmp}.txt"))
                with open(osp.join(run_directory, "config_runtime.json"), "w") as out:
                    json.dump(self.config_processed, out, skipkeys=True, indent=4, default=lambda x: None)
            else:
                run_directory = self.config_processed["chk"]
            self._run_directory = run_directory
        return self._run_directory

    @property
    def config_processed(self):
        return self.config_args
        if self._config_processed is None:
            self._config_processed = preprocess_config(self.config_args)
        return self._config_processed

    @property
    def time_it(self):
        if self._time_it is None:
            self._time_it = self.config_processed["time_debug"]
        return self._time_it
    
    @property
    def train_loader(self):
        if self._train_loader is not None:
            return self._train_loader
        
        def oversample_weights():
            if hasattr(self.ds.data, "mask"):
                # legacy oversample weights.
                # only used when training on FreeSolv-PHYSPROP-14k dataset
                has_solv_mask = self.ds.data.mask[torch.as_tensor(self.train_index), 3]
                n_total = has_solv_mask.shape[0]
                n_has_wat_solv = has_solv_mask.sum().item()
                n_no_wat_solv = n_total - n_has_wat_solv
                self.log(f"Oversampling: {n_total} molecules are in the training set")
                self.log(f"Oversampling: {n_has_wat_solv} molecules has water solv")
                self.log(f"Oversampling: {n_no_wat_solv} molecules do not have water solv")
                weights = torch.zeros_like(has_solv_mask).float()
                weights[has_solv_mask] = n_no_wat_solv
                weights[~has_solv_mask] = n_has_wat_solv
                return weights, n_total
            
            # only over-sample on the training set
            train_src_id = self.ds.data.src_id[torch.as_tensor(self.train_index)].view(-1)
            training_size = train_src_id.shape[0]
            all_src_ids = set(train_src_id.numpy().tolist())
            assert train_src_id.min() >= 0, all_src_ids
            # we want to build a lookup tensor
            assert train_src_id.max() <= 100, all_src_ids
            weight_lookup = torch.zeros(train_src_id.max() + 1).float()
            for src_id in all_src_ids:
                # the weight is inverse proportional to the number of training examples
                # the numerator is training_size instead of 1. for higher floating precision 
                # (incase the training set is too large)
                n_examples = (train_src_id == src_id).sum()
                self.log(f"Over sample: src_id=={src_id}, #examples: {n_examples}")
                weight_lookup[src_id] = (training_size / n_examples)
            weights = weight_lookup[train_src_id]
            return weights, training_size
        
        config_dict = self.config_processed
        # num_workers = os.cpu_count()
        # TODO: Distributed training num_worker problems...
        # I use cpu to debug so setting num_workers to 0 helps
        if config_dict["is_dist"] or not torch.cuda.is_available():
            num_workers = 0
        else:
            n_cpu_avail, n_cpu, num_workers = solv_num_workers()
            if config_dict["proc_in_gpu"]:
                num_workers = 0
                self.log("Since data will be preprocessed into GPU, num_workers will be set to 0")
                if n_cpu_avail > 4:
                    self.log(f"WARNING: you have requested {n_cpu_avail} CPUs for the job, but num_workers is going to be 0.")
            self.log(f"Number of total CPU: {n_cpu}")
            self.log(f"Number of available CPU: {n_cpu_avail}")
            self.log(f"Number of workers used: {num_workers}")
        
        if torch.cuda.is_available() and config_dict["is_dist"]:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.ds[torch.as_tensor(self.train_index)], shuffle=True)
            loader_kw_args = {"sampler": train_sampler, "batch_size": config_dict["batch_size"]}
        else:
            loader_kw_args = {"shuffle": not config_dict["debug_mode"], "batch_size": config_dict["batch_size"]}
            if config_dict["over_sample"]:
                assert not config_dict["dynamic_batch"]

                weights, n_total = oversample_weights()
                sampler = WeightedRandomSampler(weights=weights, num_samples=n_total)
                loader_kw_args["sampler"] = sampler
                loader_kw_args["shuffle"] = False
            elif config_dict["dynamic_batch"]:
                sampler = DynamicBatchSampler(dataset=self.ds[torch.as_tensor(self.train_index)],
                 max_num=config_dict["dynamic_batch_max_num"], shuffle=True)
                loader_kw_args["batch_sampler"] = sampler
                loader_kw_args["shuffle"] = False
                del loader_kw_args["batch_size"]
        # Tutorials:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        pin_memory = not config_dict["proc_in_gpu"]
        train_ds: Subset = Subset(self.ds, torch.as_tensor(self.train_index).tolist())
        self._train_loader = DataLoader(
            train_ds, pin_memory=pin_memory, num_workers=num_workers, **loader_kw_args)

        val_ds: Subset = Subset(self.ds, torch.as_tensor(self.val_index).tolist())
        self._val_loader = DataLoader(
            val_ds, batch_size=config_dict["valid_batch_size"],
            pin_memory=pin_memory, shuffle=False, num_workers=num_workers)
        return self._train_loader
    
    @property
    def val_loader(self):
        if self._val_loader is not None:
            return self._val_loader
        
        __ = self.train_loader
        return self._val_loader
    
    @property
    def train_index(self):
        if self._train_index is not None:
            return self._train_index
        
        config_dict = self.config_processed
        # -------------- Index file and remove specific atoms ------------ #
        train_index, val_index, test_index = self.ds.train_index, self.ds.val_index, self.ds.test_index
        if len(config_dict["remove_atom_ids"]) > 0 and config_dict["remove_atom_ids"][0] > 0:
            # remove B atom from dataset
            train_index, val_index, test_index = rm_atom(
                config_dict["remove_atom_ids"], self.ds, remove_split=('train', 'valid'),
                explicit_split=(train_index, val_index, test_index))
        if self.is_main:
            self.log('REMOVING ATOM {} FROM DATASET'.format(config_dict["remove_atom_ids"]))
            print('REMOVING ATOM {} FROM DATASET'.format(config_dict["remove_atom_ids"]))

        train_size, val_size, test_size = validate_index(train_index, val_index, test_index)

        self.log(f"train size: {train_size}")
        self.log(f"validation size: {val_size}")
        self.log(f"test size: {test_size}")

        self._train_index = train_index
        self._train_size = train_size
        self._val_index = val_index
        self._test_index = test_index

        return self._train_index
    
    @property
    def train_size(self):
        if self._train_size is None:
            # self._train_size is computed along with self.train_index
            __ = self.train_index
        return self._train_size
    
    @property
    def val_index(self):
        if self._val_index is None:
            # self._val_index is computed along with self.train_index
            __ = self.train_index
        return self._val_index
    
    @property
    def test_index(self):
        if self._test_index is None:
            # self._test_index is computed along with self.train_index
            __ = self.train_index
        return self._test_index
    
    @lazy_property
    def loss_fn(self):
        return lossfn_factory(self.config_processed)


def remove_extra_keys(data_provider, logger=None, return_extra=False):
    # These are not needed during training
    remove_names = []
    extra = {}
    example_data = data_provider.data
    for key in example_data.keys:
        if not isinstance(getattr(example_data, key), torch.Tensor):
            remove_names.append(key)
    for name in remove_names:
        if return_extra:
            extra[name] = getattr(data_provider.data, name)
        delattr(data_provider.data, name)
    if logger is not None:
        logger.info(f"These keys are deleted during training: {remove_names}")

    if return_extra:
        return data_provider, extra
    return data_provider


def dataset_from_args(args, logger=None, return_ds_args=False):
    default_kwargs = {'data_root': args["data_root"], 'pre_transform': my_pre_transform,
                      'record_long_range': True, 'type_3_body': 'B', 'cal_3body_term': True}
    data_provider_class, _kwargs = data_provider_solver(args, default_kwargs)
    _kwargs = _add_arg_from_config(_kwargs, args)
    data_provider = data_provider_class(**_kwargs)

    # if isinstance(data_provider, InMemoryDataset):
    #     data_provider = remove_extra_keys(data_provider, logger)

    if return_ds_args:
        return data_provider, _kwargs
    return data_provider


def data_provider_solver(args, default_kw_args=None, ds_key="data_provider"):
    """

    :param name_full: Name should be in a format: ${name_base}[${key}=${value}], all key-value pairs will be feed into
    data_provider **kwargs
    :param _kw_args:
    :return: Data Provider Class and kwargs
    """
    name_full = args[ds_key]
    name_base, ds_kwargs = option_solver(name_full, True, True)
    if ds_kwargs:
        if "record_name" not in args:
            # the dataset kwargs should be specified by EITHER bracket options or arguments
            assert args["dataset_name"] is None, args["dataset_name"]
            assert args["split"] is None, args["split"]
            assert args["test_name"] is None, args["test_name"]
    else:
        ds_kwargs["dataset_name"] = args["dataset_name"]
        ds_kwargs["split"] = args["split"]
        ds_kwargs["test_name"] = args["test_name"]

    if default_kw_args is None:
        default_kw_args = {}
    default_kw_args.update(ds_kwargs)
    if args["loss_metric"] == "mdn":
        default_kw_args["mdn"] = True

    default_kw_args["config_args"] = args
    if name_base == "dummy":
        assert "dataset_name", "split" in ds_kwargs
        return DummyIMDataset, default_kw_args
    elif name_base == "aux_prop":
        return AuxPropDataset, default_kw_args
    elif name_base == "pl_delta_learning":
        return PLDeltaLearningDataset, default_kw_args
    elif name_base == "casf_delta_scoring":
        return CASFSoringDeltaLearningDS, default_kw_args
    elif name_base == "casf_delta_docking":
        return CASFDockingDeltaLearningDS, default_kw_args
    elif name_base == "casf_delta_screening":
        return CASFScreeningDeltaLearningDS, default_kw_args
    elif name_base == "vs_dummy":
        return VSDummyIMDataset, default_kw_args
    elif name_base == "vs_pointer_dummy":
        return VSPointerDummyIMDataset, default_kw_args
    elif name_base == "chunk_mapper":
        return ChunkMapperDataset, default_kw_args
    elif name_base == "large":
        return LargeDataset, default_kw_args
    elif name_base == "protein_embedding":
        return ProteinEmbeddingDS, default_kw_args
    elif name_base == "pp_embedding":
        return PPEmbedDS, default_kw_args
    elif name_base == "ESMGearNetProtLig":
        from Networks.esm_gearnet.pl_dataset import ESMGearNetProtLig
        return ESMGearNetProtLig, default_kw_args
    else:
        raise ValueError('Unrecognized dataset name: {} !'.format(name_base))


def val_step_new(model, _data_loader, loss_fn: BaseLossFn, mol_lvl_detail=False, config_dict=None):
    evaluator = Evaluator(mol_lvl_detail, config_dict)
    return evaluator.compute_val_loss(model, _data_loader, loss_fn)


def train(config_dict=None, data_provider=None, explicit_split=None, ignore_valid=False, use_tqdm=False):
    trainer = Trainer(config_dict, data_provider)
    trainer.train(explicit_split, ignore_valid, use_tqdm)


def _add_arg_from_config(_kwargs, config_args):
    for attr_name in ['edge_version', 'cutoff', 'boundary_factor']:
        _kwargs[attr_name] = config_args[attr_name]
    return _kwargs


def flex_parse(add_extra_args=None):
    # set up parser and arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser = add_parser_arguments(parser)
    if add_extra_args is not None:
        parser = add_extra_args(parser)

    overwrite_comment = None
    # parse config file
    # light_overwrite_args: overwrite the args specified in the arguments that is not initialized in the config file yet
    light_overwrite_args = {}
    if len(sys.argv) == 1:
        config_name = 'config.txt'
        if os.path.isfile(config_name):
            args, unknown = parser.parse_known_args(["@" + config_name])
        else:
            raise Exception(f"Cannot find {config_name}")
    else:
        args = parser.parse_args()
        config_name = args.config_name
        overwrite_comment = args.comment
        args = vars(args)
        for key in args.keys():
            if key not in ["config_name", "comment"]:
                light_overwrite_args[key] = args[key]
        args, unknown = parser.parse_known_args(["@" + config_name])
    args.config_name = config_name
    args = vars(args)
    for key in light_overwrite_args.keys():
        if args[key] is None:
            args[key] = light_overwrite_args[key]
    if overwrite_comment is not None:
        args["comment"] = overwrite_comment

    args["local_rank"] = int(os.environ.get("LOCAL_RANK") if os.environ.get("LOCAL_RANK") is not None else 0)
    args["is_dist"] = (os.environ.get("RANK") is not None)

    args = preprocess_config(args)

    # ----------------------- Check config args: some options might be exclusive to others ----------------------- #
    if args["rmsd_threshold"] is not None:
        assert args["loss_metric"] in ["ce", "bce"] 
        assert args["target_names"] == ["within_cutoff"]
        assert args["n_output"] in [1, 2] # 1 for bce (binary cross entropy) and 2 for ce (cross entropy)
        assert not args["normalize"]
    # ------------------------------------------------------------------------------------------------------------ #

    return args

