import matplotlib.pyplot as plt
import os
import os.path as osp
import torch
import numpy as np
import seaborn as sns
from torch_scatter import scatter

from glob import glob
from sklearn.metrics import confusion_matrix


def _log(info, test_dir, end="\n"):
    # TODO: I should have used object oriented programming, but I didn't
    with open(osp.join(test_dir, "atomic_classification_analysis.log"), "a") as f:
        f.write(f"{info}{end}")


def main(test_dir, split="test"):
    # tdp: test_data_path
    tdp = glob(osp.join(test_dir, f"loss_*_{split}.pt"))
    assert len(tdp) == 1
    # td: test_data
    td = torch.load(tdp[0])

    amb = td["ATOM_MOL_BATCH"]
    if "Z_PRED" in td.keys():
        pred = td["Z_PRED"].numpy()
        label = td["ATOM_Z"].view(-1).numpy()
    else:
        raw_p = td["RAW_PRED"]
        pred = torch.argmax(raw_p, dim=-1).view(-1).numpy()
        label = td["LABEL"].view(-1).numpy()
    all_list = pred.tolist()
    all_list.extend(label.tolist())
    labels = list(set(all_list))
    conf_mat = confusion_matrix(label, pred, labels=labels)
    _log("confusion_matrix", test_dir)
    _log(labels, test_dir)
    _log(conf_mat, test_dir)
    _log(f"------{split}------", test_dir)
    sns.heatmap(conf_mat, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix on QM9_MMFF Atom Type Prediction")
    plt.savefig(osp.join(test_dir, f"confusion_matrix_{split}"))
    plt.close()

    diff = torch.as_tensor(pred != label).long()
    diff_mol = scatter(diff, amb, reduce="add", dim=0)
    num_mol = diff_mol.shape[0]
    num_error_mol = diff_mol.sum()
    _log(f"num_mols: {num_mol}", test_dir)
    _log(f"num of error mols: {num_error_mol}", test_dir)
    _log(f"mol accuracy: {1-num_error_mol/num_mol}", test_dir)
    _log("----", test_dir)
    _log(f"num_atoms: {diff.shape[0]}", test_dir)
    _log(f"num of error atoms: {diff.sum()}", test_dir)
    _log(f"atom accuracy: {1-diff.sum()/diff.shape[0]}", test_dir)


if __name__ == '__main__':
    main("../../raw_data/exp200-400/exp393_run_2021-10-14_073422/exp393_test_2021-10-20_162845", "val")
    main("../../raw_data/exp200-400/exp393_run_2021-10-14_073422/exp393_test_2021-10-20_162845", "test")
