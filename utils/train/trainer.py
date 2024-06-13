import argparse
import glob
import json
import logging
import math
import os
import os.path as osp
import time
from collections import OrderedDict
import re
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.cuda
import torch.utils.data
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader, DynamicBatchSampler
from tqdm import tqdm

from Networks.PhysDimeNet import PhysDimeNet
from utils.DataPrepareUtils import my_pre_transform
from utils.LossFn import lossfn_factory
from utils.data.DummyIMDataset import AuxPropDataset, DummyIMDataset, VSDummyIMDataset, VSPointerDummyIMDataset
from utils.data.data_utils import data_to_device, get_lig_natom
from utils.data.delta_learning_ds import CASFSoringDeltaLearningDS, CASFDockingDeltaLearningDS, CASFScreeningDeltaLearningDS, PLDeltaLearningDataset
from utils.data.LargeDataset import LargeDataset
from utils.Optimizers import EmaAmsGrad
from utils.data.prot_embedding_ds import PPEmbedDS, ProteinEmbeddingDS, ProteinEmbeddingFlyDS
from utils.data.vs_im_datasets import ChunkMapperDataset
from utils.eval.evaluator import Evaluator
from utils.tags import tags
from utils.utils_functions import floating_type, lazy_property, get_lr, get_n_params, \
    atom_mean_std, remove_handler, option_solver, fix_model_keys, get_device, process_state_dict, \
    validate_index, non_collapsing_folder, solv_num_workers
from utils.configs import Config, schema


class Trainer:
    def __init__(self, cfg, data_provider=None):
        self.cfg: Config = cfg

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

    @lazy_property
    def evaluator(self):
        return Evaluator(False, self.cfg)

    def train(self, explicit_split=None, use_tqdm=False):
        if explicit_split is not None:
            torch.save(explicit_split, osp.join(self.run_directory, "runtime_split.pt"))
            if isinstance(explicit_split, tuple):
                self._train_index, self._val_index, self._test_index = explicit_split
            elif isinstance(explicit_split, dict):
                self._train_index, self._val_index, self._test_index = \
                    explicit_split["train_index"], explicit_split["valid_index"], explicit_split["test_index"]
            else:
                raise ValueError(f"cannot understand split: {explicit_split}")
        
        # ------------------- variable set up ---------------------- #
        cfg = self.cfg

        self.log("used dataset: {}".format(self.ds.processed_file_names))

        # ------------------- Setting up model and optimizer ------------------ #
        mean_atom, std_atom = self.get_norms()
        train_model = PhysDimeNet(cfg=cfg, ds=self.ds, energy_shift=mean_atom, energy_scale=std_atom)
        train_model = train_model.to(get_device())

        # ---------------------- restore pretrained model ------------------------ #
        def get_pretrained_paths() -> Tuple[str, str, Optional[str]]:
            model_chk = cfg.model["use_trained_model"]
            if not osp.isdir(model_chk):
                # the pth file is directly provided.
                return model_chk, model_chk, None

            # the pth file is infered from the directory
            trained_model_dir = glob.glob(model_chk)
            assert len(trained_model_dir) == 1, f"Zero or multiple trained folder: {trained_model_dir}"
            trained_model_dir = trained_model_dir[0]
            cfg.model["use_trained_model"] = trained_model_dir
            self.log('using trained model: {}'.format(trained_model_dir))
            train_model_path = osp.join(trained_model_dir, 'training_model.pt')
            if cfg.model.ft_discard_training_model or not osp.exists(train_model_path):
                train_model_path = osp.join(trained_model_dir, 'best_model.pt')
            val_model_path = osp.join(trained_model_dir, 'best_model.pt')
            return train_model_path, val_model_path, trained_model_dir
        
        if cfg.model["use_trained_model"]:
            train_model_path, val_model_path, trained_model_dir = get_pretrained_paths(cfg)
            train_dict = torch.load(train_model_path, map_location=get_device())
            train_dict = fix_model_keys(train_dict)
            train_dict = process_state_dict(train_dict, cfg, self.logger)

            incompatible_keys = train_model.load_state_dict(state_dict=train_dict, strict=False)

            self.log(f"---------vvvvv incompatible keys in {train_model_path} vvvvv---------")
            for name in ["missing_keys", "unexpected_keys"]:
                self.log(name.upper()+" :")
                for key in getattr(incompatible_keys, name):
                    self.log(key)
                self.log("*"*10)

            shadow_dict = torch.load(val_model_path, map_location=get_device())
            shadow_dict = process_state_dict(fix_model_keys(shadow_dict), cfg, self.logger)
        else:
            shadow_dict = None
            trained_model_dir = None
        self.train_model = train_model

        # optimizers
        assert cfg.training.optimizer.split('_')[0] == 'emaAms'
        optimizer = EmaAmsGrad(train_model, lr=cfg.training.learning_rate,
                                ema=float(cfg.training.optimizer.split('_')[1]),
                                shadow_dict=shadow_dict, params=self.get_optim_params(train_model),
                                use_buffers=cfg.training["swa_use_buffers"], run_dir=self.run_directory)
        self.optimizer: EmaAmsGrad = optimizer

        # schedulers
        scheduler_kw_args = option_solver(cfg.training.scheduler, type_conversion=True)
        scheduler_base = cfg.training.scheduler.split("[")[0]
        self.scheduler_base = scheduler_base
        if scheduler_base == "StepLR":
            step_per_epoch = 1. * self.train_size / cfg.training.batch_size
            decay_steps = math.ceil(scheduler_kw_args["decay_epochs"] * step_per_epoch)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_steps, gamma=0.1)
        elif scheduler_base == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kw_args)
        else:
            raise ValueError('Unrecognized scheduler: {}'.format(cfg.training.scheduler))

        if trained_model_dir and (not cfg.model.reset_optimizer):
            if os.path.exists(os.path.join(trained_model_dir, "best_model_optimizer.pt")):
                optimizer.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_optimizer.pt"),
                                                     map_location=get_device()))
            if os.path.exists(os.path.join(trained_model_dir, "best_model_scheduler.pt")):
                scheduler.load_state_dict(torch.load(os.path.join(trained_model_dir, "best_model_scheduler.pt"),
                                                     map_location=get_device()))
        self.scheduler = scheduler

        if cfg.model["use_trained_model"]:
            # protect LR been reduced
            step_per_epoch = 1. * self.train_size / cfg.training.batch_size
            ema_avg = 1 / (1 - float(cfg.training["optimizer"].split('_')[1]))
            ft_protection = math.ceil(ema_avg / step_per_epoch)
            ft_protection = min(30, ft_protection)
        else:
            ft_protection = 0

        self.log(f"Fine tune protection: {ft_protection} epochs. ")

        # --------------------- Printing meta data ---------------------- #
        if get_device().type == 'cuda':
            self.log('Hello from device : ' + torch.cuda.get_device_name(get_device()))
            self.log("Cuda mem allocated: {:.2f} MB".format(torch.cuda.memory_allocated(get_device()) * 1e-6))
        else:
            print("You are not using a GPU")

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

        # ---------------------- Training ----------------------- #
        self.log('start training...')

        shadow_net = optimizer.shadow_model
        val_res = self.evaluator(shadow_net, self.val_loader, self.loss_fn)

        valid_info_dict = OrderedDict(
            {"epoch": 0, "train_loss": -1, "valid_loss": val_res["loss"], "delta_time": time.time() - self.t0})
        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                valid_info_dict[key] = val_res[key]
                valid_info_dict.move_to_end(key)
        self.update_df(valid_info_dict)

        # use np.inf instead of val_res["loss"] for proper transfer learning behaviour
        self.best_loss = np.inf

        last_epoch = pd.read_csv(osp.join(self.run_directory, "loss_data.csv"), header="infer").iloc[-1]["epoch"]
        last_epoch = int(last_epoch.item())
        self.log('Init lr: {}'.format(get_lr(optimizer)))

        self.step = 0

        self.log("Setup complete, training starts...")

        for epoch in range(last_epoch, last_epoch + cfg.training["num_epochs"]):

            # Early stop when learning rate is too low
            this_lr = get_lr(optimizer)
            if cfg.training.stop_low_lr and \
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
                    train_loss += self.train_step(train_model, data_batch=data) * this_size / self.train_size
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
                if cfg.training.eval_per_step is not None and (self.step % cfg.training.eval_per_step == 0):
                    # evaluate every k steps specified in --eval_per_step
                    self.eval_step(train_loss, epoch)

            if cfg.training.eval_per_step is None:
                # By default, evaluate every epoch
                self.eval_step(train_loss, epoch)
            if scheduler_base in tags.step_per_epoch and (epoch - last_epoch) >= ft_protection:
                scheduler.step(metrics=val_res["loss"])

        remove_handler(self.logger)
        meta = {"run_directory": self.run_directory}
        return meta

    def eval_step(self, train_loss, epoch: int):
        # ---------------------- Evaluation step: validation, save model, print meta ---------------- #
        if not self.is_main:
            return
        this_lr = get_lr(self.optimizer)
        loss_fn = self.loss_fn
        cfg = self.cfg

        self.log('step {} ended, learning rate: {} '.format(self.step, this_lr))
        shadow_net = self.optimizer.shadow_model

        val_res = self.evaluator(shadow_net, self.val_loader, loss_fn)

        valid_info_dict = {"step": self.step, "epoch": epoch, "train_loss": train_loss, 
                           "valid_loss": val_res["loss"], "delta_time": time.time() - self.t0}

        for key in val_res.keys():
            if key != "loss" and not isinstance(val_res[key], torch.Tensor):
                valid_info_dict[key] = val_res[key]

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
            if self.early_stop_count == cfg.training.early_stop:
                self.log('early stop at epoch {}.'.format(epoch))
                return True
        return False

    def get_norms(self):
        if not self.cfg.model.normalization.normalize:
            return 0., 1.
        train_index = self.train_index
        # skip scale and shift calculation if transfer learning
        if self.cfg.model.use_trained_model and not self.cfg.model.reset_scale_shift:
            return 0. ,1.
        
        # scale and shift not used in MDN loss
        if self.cfg.training.loss_fn.loss_metric == "mdn":
            return 0. ,1.

        if self.cfg.training.loss_fn.action in ["names", "names_and_QD"]:
            mean_atom = []
            std_atom = []
            for name in self.cfg.training.loss_fn.target_names:
                N_matrix = self.ds.data.N
                this_mean, this_std = atom_mean_std(getattr(self.ds.data, name), N_matrix, train_index)
                mean_atom.append(this_mean)
                std_atom.append(this_std)
            if self.cfg.training.loss_fn.action == "names_and_QD":
                # the last dimension is for predicting atomic charge
                mean_atom.append(0.)
                std_atom.append(1.)
            mean_atom = torch.as_tensor(mean_atom)
            std_atom = torch.as_tensor(std_atom)
        elif self.cfg.training.loss_fn.action == "names_atomic":
            mean_atom = []
            std_atom = []
            for name in self.cfg.training.loss_fn.target_names:
                this_atom_prop: torch.Tensor = getattr(self.ds.data, name)
                mean_atom.append(torch.mean(this_atom_prop.type(floating_type)).item())
                std_atom.append(torch.std(this_atom_prop.type(floating_type)).item())
            mean_atom = torch.as_tensor(mean_atom)
            std_atom = torch.as_tensor(std_atom)
        else:
            raise ValueError("Invalid action: {}".format(self.cfg.training.loss_fn.action))

        return mean_atom, std_atom

    def train_step(self, model, data_batch):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # self.t0 = time.time()
        with torch.autograd.set_detect_anomaly(False):
            model.train()
            if self.cfg.model.mdn.mdn_freeze_bn:
                # turn off the batch normalization in the MDN layer for proper fine-tune behaviour
                model.mdn_layer.MLP[1].eval()
            self.optimizer.zero_grad()

            model_out = model(data_batch)
            loss = self.loss_fn(model_out, data_batch, True) + model_out["nh_loss"]
            loss.backward()

        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.max_norm, 
                                           error_if_nonfinite=self.cfg.training.error_if_nonfinite)
        except RuntimeError as e:
            if "is non-finite, so it cannot be clipped" not in str(e):
                raise e
            # the NaN gradient error happens when training the cross_mdn_pkd loss
            logging.warning("SKIPPING batch: " + str(e))
            return 0.
        self.optimizer.step()
        if self.scheduler_base in tags.step_per_step:
            self.scheduler.step()

        result_loss = loss.data[0].cpu().item()

        return result_loss

    def get_optim_params(self, model):
        if self.cfg.training.ft_lr_factor is None:
            return model.parameters()

        normal_lr_parms = []
        reduced_lr_parms = []
        reduced_lr = self.cfg.training.learning_rate*self.cfg.training.ft_lr_factor
        freeze: bool = (self.cfg.training.ft_lr_factor == 0)
        normal_lr_ptns = [re.compile(ptn) for ptn in self.cfg.training.normal_lr_ptn]
        lower_lr_ptns = [re.compile(ptn) for ptn in self.cfg.training.lower_lr_ptn]
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
            self._ds, self._ds_args = dataset_from_args(self.cfg, self.logger, return_ds_args=True)
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
        if self._logger is None:
            # --------------------- Logger setup ---------------------------- #
            # first we want to remove previous logger step up by other programs
            # we sometimes encounter issues and the logger file doesn't show up
            log_tmp = logging.getLogger()
            remove_handler(log_tmp)
            logging.basicConfig(filename=osp.join(self.run_directory, "training.log"),
                                format='%(asctime)s %(message)s', filemode='w')
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            self._logger = logger
        return self._logger

    @property
    def run_directory(self):
        if self._run_directory is None:
            # ----------------- set up run directory -------------------- #
            folder_prefix = self.cfg["folder_prefix"]
            folder_prefix_base = osp.basename(folder_prefix)
            run_directory = non_collapsing_folder(folder_prefix)
            with open(osp.join(run_directory, f"config-{folder_prefix_base}.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            with open(osp.join(run_directory, "config_runtime.json"), "w") as out:
                json.dump(self.cfg, out, skipkeys=True, indent=4, default=lambda x: None)
            self._run_directory = run_directory
        return self._run_directory
    
    @property
    def train_loader(self):
        if self._train_loader is not None:
            return self._train_loader
        
        def oversample_weights():
            # only over-sample on the training set
            train_file_handle: List[str] = [self.ds.data.file_handle[i] for i in self.train_index]
            train_src_id = [0 if fl.startswith("nonbinders.") else 1 for fl in train_file_handle]
            train_src_id = torch.as_tensor(train_src_id)
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
        
        n_cpu_avail, n_cpu, num_workers = solv_num_workers()
        if self.cfg.data.proc_in_gpu:
            num_workers = 0
            self.log("Since data will be preprocessed into GPU, num_workers will be set to 0")
            if n_cpu_avail > 4:
                self.log(f"WARNING: you have requested {n_cpu_avail} CPUs for the job, but num_workers is going to be 0.")
        self.log(f"Number of total CPU: {n_cpu}")
        self.log(f"Number of available CPU: {n_cpu_avail}")
        self.log(f"Number of workers used: {num_workers}")
        
        loader_kw_args = {"shuffle": not self.cfg["debug_mode"], "batch_size": self.cfg["batch_size"]}
        if self.cfg.data.over_sample:
            assert not self.cfg.data.dynamic_batch

            weights, n_total = oversample_weights()
            sampler = WeightedRandomSampler(weights=weights, num_samples=n_total)
            loader_kw_args["sampler"] = sampler
            loader_kw_args["shuffle"] = False
        elif self.cfg.data.dynamic_batch:
            sampler = DynamicBatchSampler(dataset=self.ds[torch.as_tensor(self.train_index)],
                max_num=self.cfg.data.dynamic_batch_max_num, shuffle=True)
            loader_kw_args["batch_sampler"] = sampler
            loader_kw_args["shuffle"] = False
            del loader_kw_args["batch_size"]
        # Tutorials:
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        pin_memory = not self.cfg.data.proc_in_gpu
        train_ds: Subset = Subset(self.ds, torch.as_tensor(self.train_index).tolist())
        self._train_loader = DataLoader(
            train_ds, pin_memory=pin_memory, num_workers=num_workers, **loader_kw_args)

        val_ds: Subset = Subset(self.ds, torch.as_tensor(self.val_index).tolist())
        self._val_loader = DataLoader(
            val_ds, batch_size=self.cfg.training.valid_batch_size,
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
        
        # -------------- Index file and remove specific atoms ------------ #
        train_index, val_index, test_index = self.ds.train_index, self.ds.val_index, self.ds.test_index

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
        return lossfn_factory(self.cfg)


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

    if return_ds_args:
        return data_provider, _kwargs
    return data_provider


def data_provider_solver(cfg: Config, default_kw_args=None):
    """

    :param name_full: Name should be in a format: ${name_base}[${key}=${value}], all key-value pairs will be feed into
    data_provider **kwargs
    :param _kw_args:
    :return: Data Provider Class and kwargs
    """
    name_full = cfg.data.data_provider
    data_cfg = cfg.data
    name_base, ds_kwargs = option_solver(name_full, True, True)
    if ds_kwargs:
        if "record_name" not in cfg:
            # the dataset kwargs should be specified by EITHER bracket options or arguments
            assert data_cfg["dataset_name"] is None, data_cfg["dataset_name"]
            assert data_cfg["split"] is None, data_cfg["split"]
            assert data_cfg["test_name"] is None, data_cfg["test_name"]
    else:
        ds_kwargs["dataset_name"] = data_cfg["dataset_name"]
        ds_kwargs["split"] = data_cfg["split"]
        ds_kwargs["test_name"] = data_cfg["test_name"]

    if default_kw_args is None:
        default_kw_args = {}
    default_kw_args.update(ds_kwargs)
    if cfg.training.loss_fn.loss_metric == "mdn":
        default_kw_args["mdn"] = True

    default_kw_args["cfg"] = cfg
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
    elif name_base == "protein_embedding_fly":
        return ProteinEmbeddingFlyDS, default_kw_args
    elif name_base == "pp_embedding":
        return PPEmbedDS, default_kw_args
    elif name_base == "ESMGearNetProtLig":
        from Networks.esm_gearnet.pl_dataset import ESMGearNetProtLig
        return ESMGearNetProtLig, default_kw_args
    else:
        raise ValueError('Unrecognized dataset name: {} !'.format(name_base))


def train(config_dict=None, data_provider=None, explicit_split=None, ignore_valid=False, use_tqdm=False):
    trainer = Trainer(config_dict, data_provider)
    trainer.train(explicit_split, ignore_valid, use_tqdm)


def parse_args():
    # set up parser and arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name")
    args = parser.parse_args()
    config_name = args.config_name
    config = OmegaConf.load(config_name)
    config = OmegaConf.merge(schema, config)
    return config

