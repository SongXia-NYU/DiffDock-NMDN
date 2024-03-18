import torch
import torch.nn as nn

from torch_geometric.data import Batch
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config, load_state_dict
# the following import line is nessasary to get it registered 
from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from ocpmodels.models.equiformer_v2.so3 import SO3_Embedding
from utils.utils_functions import get_device, floating_type

# A wrapper class to use EquiformerV2 by OCP
# Differences:
# 1. use_pbc is disabled.
# 2. unless specified, only the atom embedding is needed.
class EquiformerV2(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.pred_energy: bool = cfg["equiformer_v2_for_energy"]
        self.narrow_embed: bool = cfg["equiformer_v2_narrow_embed"]
        self.load_model()
        self.load_ckpt(cfg)
        

    def load_model(self):
        model_cfg_file = "/scratch/sx801/scripts/ocp-1205/configs/s2ef/all/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.valonly.yml"
        model_cfg, __, __ = load_config(model_cfg_file)
        model_cfg = model_cfg["model"]

        model_name = model_cfg.pop("name")
        model_attributes = model_cfg
        bond_feat_dim = model_attributes.get(
            "num_gaussians", 50
        )
        num_targets = 3 if self.pred_energy else 1
        model: EquiformerV2_OC20 = registry.get_model_class(model_name)(
            None, bond_feat_dim, num_targets, **model_attributes
        ).to(get_device())
        # use_pbc is disabled here since the ligand is not in crystal structure
        model.use_pbc = False
        self.model: EquiformerV2_OC20 = model

    def load_ckpt(self, cfg: dict):
        if cfg["equiformer_v2_ckpt"] is None:
            raise ValueError("Checkpoint file not found. You do not want to train it from scratch")
        checkpoint = torch.load(
                    cfg["equiformer_v2_ckpt"], map_location=get_device()
                )
        
        # The following code is adapted from OCP.
        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        if self.pred_energy:
            # modify the checkpoint parameters: the pre-trained model was trained on a single task
            # the energy prediction task contains three tasks (gas, water and octanol)
            new_dict["energy_block.so3_linear_2.weight"] = torch.concat([new_dict["energy_block.so3_linear_2.weight"]]*3, dim=1)
            new_dict["energy_block.so3_linear_2.bias"] = torch.concat([new_dict["energy_block.so3_linear_2.bias"]]*3, dim=0)
        load_state_dict(self.model, new_dict, strict=True)


    def forward(self, runtime_vars: dict):
        if self.pred_energy:
            data_batch = self.get_data_batch(runtime_vars)
            energy, force = self.model.forward(data_batch, return_embedding=False)
            runtime_vars["mol_prop"] = energy
            return runtime_vars
        
        # pred embedding only
        return self.forward_embedding(runtime_vars)


    def forward_embedding(self, runtime_vars: dict):
        data_batch = self.get_data_batch(runtime_vars)
        # [-1, 25, 128]
        atom_embedding: SO3_Embedding = self.model.forward(data_batch, return_embedding=True)
        if self.narrow_embed:
            # Only select the first embedding. [-1, 25, 128] -> [-1, 128]
            atom_embedding = atom_embedding.embedding.narrow(1, 0, 1).squeeze(1)
        else:
            # [-1, 25, 128] -> [-1, 25*128]
            atom_embedding = atom_embedding.embedding.view(-1, 25 * 128)
        runtime_vars["vi"] = atom_embedding
        return runtime_vars


    def get_data_batch(self, runtime_vars: dict) -> Batch:
        # preprocess data batch into a format that is readable by EquiformerV2
        data_batch: Batch = runtime_vars["data_batch"]
        data_batch.pos = data_batch.R.type(floating_type)
        data_batch.natoms = data_batch.N.type(floating_type)
        data_batch.atomic_numbers = data_batch.Z.type(floating_type)
        data_batch.batch = data_batch.atom_mol_batch
        return data_batch
    