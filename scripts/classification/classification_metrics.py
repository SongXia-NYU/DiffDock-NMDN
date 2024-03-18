from sklearn.metrics import roc_auc_score
import torch
import numpy as np
from glob import glob
import os.path as osp


def classification_metrics(folders):
    for folder in glob(folders):
        for split in ["val", "test"]:
            test_data_p = glob(osp.join(folder, "*_test_*", f"loss_*_{split}.pt"))
            assert len(test_data_p) == 1
            test_data = torch.load(test_data_p[0])
            label = test_data["LABEL"]
            prob = test_data["RAW_PRED"][:, -1]
            auc_roc = roc_auc_score(label, prob)
            with open(osp.join(osp.dirname(test_data_p[0]), "classification_test.txt"), "a") as f:
                f.write(f"{split.upper()}_AUC_ROC: {auc_roc}\n")


if __name__ == '__main__':
    classification_metrics("../../tmp/exp*_run_*")
