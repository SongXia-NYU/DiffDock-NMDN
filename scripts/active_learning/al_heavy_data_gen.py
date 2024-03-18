import argparse
import glob
import os
import os.path as osp

import pandas as pd
import torch
from deprecated import deprecated
from torch_geometric.loader import DataLoader

from Networks.PhysDimeNet import PhysDimeNet
from active_learning import ALTrainer
from utils.train.trainer import val_step_new
from utils.utils_functions import preprocess_config, get_device, floating_type


class ALHeavyDataGen(ALTrainer):
    @deprecated("Use ALTrainer.validate_selection_cycle instead.")
    def gen_uncertainty_detail(self):
        # Go through the whole al process, record run-time uncertainty data in each selection step
        for n_cycle, n_heavy in enumerate(self.n_heavy_list):
            print(f"working on cycle={n_cycle}")
            this_info = {}
            this_root = osp.join(self.al_folder, "heavy_data")
            os.makedirs(this_root, exist_ok=True)

            prev_folders = glob.glob(f"{self.folder_format(n_cycle=n_cycle-1)}_run_*")
            assert len(prev_folders) == 1
            prev_folder = prev_folders[0]
            best_model_sd = torch.load(osp.join(prev_folder, "best_model.pt"), map_location=get_device())
            processed_arg = preprocess_config(self.args)
            model = PhysDimeNet(cfg=processed_arg, **processed_arg).to(get_device()).type(floating_type)
            model.load_state_dict(best_model_sd)

            candidate_index = self.get_candidate(n_heavy, n_cycle, is_training=False)
            data_loader = DataLoader(
                self.data_provider[torch.as_tensor(candidate_index)], batch_size=self.args["valid_batch_size"],
                pin_memory=torch.cuda.is_available(), shuffle=False)
            result = val_step_new(model, data_loader, self.loss_fn, mol_lvl_detail=True, lightweight=True)
            uncertainty = result["UNCERTAINTY"].view(-1)
            assert len(uncertainty) == len(candidate_index)
            sort_idx = uncertainty.argsort(descending=True)

            this_info["uncertainty"] = uncertainty[sort_idx].tolist()
            this_info["log_uncertainty"] = torch.log(uncertainty[sort_idx]).tolist()
            this_info["idx_in_dataset"] = candidate_index[sort_idx].tolist()
            info_df = pd.DataFrame(this_info)
            info_df.to_csv(osp.join(this_root, f"uncertainty_cycle_{n_cycle}.csv"), index=False)


def main(folder_path):
    data_generator = ALHeavyDataGen(chk=folder_path)
    data_generator.gen_uncertainty_detail()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk")
    args = parser.parse_args()
    main(args.chk)
