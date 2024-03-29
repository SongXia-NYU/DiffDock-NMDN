import torch
import os
import os.path as osp

dst_root = "/scratch/sx801/scripts/results"
src = "/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/best_model.pt"
dst_folder = f"{dst_root}/exp_pl_534for_pl538"
os.makedirs(dst_folder, exist_ok=True)

state_dict = torch.load(src, map_location="cpu")

key = "module.main_module_list.4.lig_prot_layer.readout.0.weight"
state_dict[key] = state_dict[key][:, :160]

torch.save(state_dict, osp.join(dst_folder, "best_model.pt"))
