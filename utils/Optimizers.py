import time

import torch
import copy
import os.path as osp

from torch.nn import Module
from torch.optim.swa_utils import AveragedModel

from utils.time_meta import record_data
from utils.utils_functions import get_device


class EmaAmsGrad(torch.optim.Adam):
    def __init__(self, training_model: torch.nn.Module, lr=1e-3, betas=(0.9, 0.99),
                 eps=1e-8, weight_decay=0, ema=0.999, shadow_dict=None, params=None, use_buffers=False, start_step=0, run_dir=None):
        if params is None:
            params = training_model.parameters()
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad=True)
        self.current_step = 0
        self.ema = ema
        self.training_model = training_model
        self.start_step = start_step
        self.use_buffers = use_buffers
        self.run_dir = run_dir

        self.disabled = (ema < 0)
        self._shadow_model = None
        if self.shadow_model_available() and start_step == 0:
            # SWA is enabled at the beginning, shadow_model will be initialized by provided shadow_dict
            self.init_shadow_model(shadow_dict)

    def step(self, closure=None):
        loss = super().step(closure)

        if self.shadow_model_available():
            if self.start_step != 0 and (self.current_step == self.start_step):
                # If SWA is enabled during training, both shadow model and training model will be initilized by best model
                best_model_dict = torch.load(osp.join(self.run_dir, "best_model.pt"))
                self.init_shadow_model(best_model_dict)
                self.training_model.load_state_dict(best_model_dict)
            self._shadow_model.update_parameters(self.training_model)
        self.current_step += 1
        return loss

    @property
    def shadow_model(self):
        if not self.shadow_model_available():
            return self.training_model
        return self._shadow_model

    def shadow_model_available(self):
        if self.disabled:
            return False
        return self.current_step >= self.start_step

    def init_shadow_model(self, shadow_dict):
        assert self._shadow_model is None
        ema = self.ema
        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            if not model_parameter.requires_grad:
                return model_parameter
            return ema * averaged_model_parameter + (1 - ema) * model_parameter

        self._shadow_model = AveragedModel(self.training_model, device=get_device(), use_buffers=self.use_buffers, avg_fn=avg_fn)
        if self._shadow_model.n_averaged == 0 and shadow_dict is not None:
            self._shadow_model.module.load_state_dict(shadow_dict, strict=False)
            self._shadow_model.n_averaged += 1


class MySGD(torch.optim.SGD):
    """
    my wrap of SGD for compatibility issues
    """

    def __init__(self, model, *args, **kwargs):
        self._shadow_model = model
        super(MySGD, self).__init__(model.parameters(), *args, **kwargs)

    def step(self, closure=None):
        return super(MySGD, self).step(closure)
