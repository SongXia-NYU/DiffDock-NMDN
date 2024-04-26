import math
from typing import Dict, Union
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from utils.LossFn import BaseLossFn
from utils.configs import Config
from utils.data.data_utils import data_to_device
from utils.tags import tags

class Evaluator:
    def __init__(self, mol_lvl_detail=False, cfg: Config=None) -> None:
        self.mol_lvl_detail: bool = mol_lvl_detail
        self.cfg = cfg

    def init_vars(self):
        # number of molecule validated in current batch. It is used to record proper batch information
        self.valid_size: int = 0
        # This is used in flexible training where only part of the properties are present in molecules
        self.flex_size: dict = {}
        # number of batchs in the validation loader
        self.n_batchs = 0
        self.loss: float = 0.
        self.loss_detail: dict = None

    def __call__(self, *args, **kwds):
        return self.compute_val_loss(*args, **kwds)

    @torch.no_grad()
    def compute_val_loss(self, model: nn.Module, data_loader: DataLoader, loss_fn: BaseLossFn):
        self.init_vars()
        model.eval()
        for i, val_data in enumerate(data_loader):
            val_data = data_to_device(val_data)
            model_out = model(val_data)
            aggr_loss, batch_detail = loss_fn(model_out, val_data, False, True, mol_lvl_detail=self.mol_lvl_detail)
            if self.cfg.model.mdn.nmdn_eval:
                aggr_loss = -batch_detail["MDN_LOGSUM_DIST2_REFDIST2"].sum().cpu()
            self.record_detail(aggr_loss, batch_detail)
            self.n_batchs += 1
            self.valid_size += batch_detail["n_units"]
            
        self.compute_average()
        return self.loss_detail
    
    def record_detail(self, aggr_loss: torch.Tensor, batch_detail: Dict[str, torch.Tensor]):
        # n_units is the batch size when predicting mol props but number of atoms when predicting atom props.
        n_units = batch_detail["n_units"]
        self.loss += aggr_loss.item() * n_units
        if self.loss_detail is None:
            self.loss_detail = self.get_init_loss_detail(batch_detail)

        for key in batch_detail:
            if tags.val_avg(key):
                if "flex_units" in batch_detail.keys():
                    # This is used in flexible training where only part of the properties are present in molecules
                    prop_name = key.split("_")[-1]
                    batch_detail[key] += batch_detail[key] * batch_detail["flex_units"][prop_name]
                    if key not in self.flex_size.keys():
                        self.flex_size[key] = 0
                    self.flex_size[key] += batch_detail["flex_units"][prop_name]
                    continue
                if key == "mdn_hist":
                    self.loss_detail[key] = self.hist_add(self.loss_detail[key], batch_detail[key])
                    continue
                self.loss_detail[key] += batch_detail[key] * n_units
            elif tags.val_concat(key) and key not in ["ATOM_MOL_BATCH"]:
                self.loss_detail[key].append(batch_detail[key])
            elif key == "ATOM_MOL_BATCH":
                self.loss_detail[key].append(batch_detail[key] + self.valid_size)

    def get_init_loss_detail(self, batch_detail: Dict[str, torch.Tensor]):
        # -----init------ #
        loss_detail = {}

        for key in batch_detail.keys():
            if tags.val_avg(key):
                loss_detail[key] = 0.
            elif tags.val_concat(key):
                if key == "atom_embedding": continue
                loss_detail[key] = []
        return loss_detail
    
    def compute_average(self):
        loss_detail = self.loss_detail
        loss_detail["n_units"] = self.valid_size
        for key in self.flex_size:
            loss_detail[f"n_units_{key}"] = self.flex_size[key]

        self.loss /= self.valid_size
        # Stacking if/else like hell
        for key in list(loss_detail.keys()):
            if tags.val_avg(key):
                if key in self.flex_size.keys():
                    loss_detail[key] /= self.flex_size[key] if self.flex_size[key] != 0 else None
                    continue
                if key == "mdn_hist":
                    loss_detail[key] /= self.n_batchs
                loss_detail[key] /= self.valid_size
            elif tags.val_concat(key):
                loss_detail[key] = torch.cat(loss_detail[key], dim=0)

            if key.startswith("MSE_"):
                if loss_detail[key] is not None:
                    loss_detail[f"RMSE_{key.split('MSE_')[-1]}"] = math.sqrt(loss_detail[key])
                else:
                    loss_detail[f"RMSE_{key.split('MSE_')[-1]}"] = None
        loss_detail["loss"] = self.loss

    @staticmethod
    def hist_add(a: Union[torch.Tensor, float], b: torch.Tensor):
        if isinstance(a, float):
            assert a == 0., str(a)
            return b
        # add histagram values
        if a.shape[0] < b.shape[0]:
            b[:a.shape[0]] += a
            return b
        a[:b.shape[0]] += b
        return a



