import os
import os.path as osp
import sys

import torch
import torch.nn as nn
from easydict import EasyDict

from torchdrug import data, utils, core

from utils.utils_functions import get_device
# /scratch/sx801/scripts/ESM-GearNet
sys.path.append("/scratch/sx801/scripts/ESM-GearNet")
from util import load_config
# this import is required to register `FusionNetwork` aka the model
from gearnet import model

class ESMGearnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        esm_gearnet_cofig_file: str = osp.join(osp.dirname(__file__), "esm_gearnet.yaml")
        egearnet_cfg: EasyDict = load_config(esm_gearnet_cofig_file, {})

        task = core.Configurable.load_config_dict(egearnet_cfg.task)
        self.graph_construction_model = task.graph_construction_model
        self.model = task.model
        # loading model checkpoint
        model_dict = torch.load(egearnet_cfg.model_checkpoint, map_location=get_device())
        self.model.load_state_dict(model_dict)

    def forward(self, runtime_vars: dict):
        protein = runtime_vars["data_batch"]["graph"]
        protein = self.graph_construction_model(protein)
        res: dict = self.model(protein, protein.node_feature.float())
        runtime_vars["prot_embed"] = res["node_feature"]
        runtime_vars["prot_pos"] = protein.node_position
        return runtime_vars