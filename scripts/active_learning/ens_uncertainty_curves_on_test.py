import os
from glob import glob
import os.path as osp
import torch

from scripts.uncertainty_curves import uncertainty_curves_ens


def ens_uncertainty_curves_on_test(root_folder):
    os.makedirs(osp.join(root_folder, "ens_test_uncertainty_curves"), exist_ok=True)
    folders = glob(osp.join(root_folder, "exp*_cycle_*_run_*"))
    cycle_to_folders = {}
    for f in folders:
        cycle = _get_cycle(f)
        if cycle not in cycle_to_folders:
            cycle_to_folders[cycle] = []
        cycle_to_folders[cycle].append(f)
    n_ens = len(cycle_to_folders[-1])

    for cycle in cycle_to_folders:
        # if cycle != 10:
        #     print(f"Skipping cycle: {cycle}")
        #     continue
        this_folders = cycle_to_folders[cycle]
        assert n_ens == len(this_folders), f"{cycle_to_folders}"
        predictions = []
        tgt = None
        for f in this_folders:
            this_loss = torch.load(glob(osp.join(f, f"exp*_cycle_{cycle}_test_*", "loss_*_test.pt"))[0])
            predictions.append(this_loss["PROP_PRED"].unsqueeze(-1))
            if tgt is None:
                tgt = this_loss["PROP_TGT"]
        predictions = torch.cat(predictions, dim=-1)
        uncertainty_curves_ens(predictions, tgt, osp.join(root_folder, "ens_test_uncertainty_curves"), cycle, n_ens)


def _get_cycle(folder):
    str_cycle = osp.basename(folder).split("_run_")[0].split("_cycle_")[-1]
    return int(str_cycle)


if __name__ == '__main__':
    ens_uncertainty_curves_on_test("../../raw_data/exp400-500/exp406_active_ALL_2021-11-22_111836")
