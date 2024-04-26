"""
Deeper analysis of tested results like error distribution, etc..
Only works on LargeDataset
"""
from glob import glob
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import os.path as osp

import torch

from utils.eval.TestedFolderReader import TestedFolderReader
from utils.utils_functions import error_message


class TestResultAnalyzer(TestedFolderReader):
    def __init__(self, folder_name, tested_folder_name, root="../results"):
        super().__init__(folder_name, tested_folder_name, root)

    def run(self):
        file_locator = torch.load(osp.join(self.cfg["data_root"], self.ds_options["file_locator"]))
        split_index = torch.load(osp.join(self.cfg["data_root"], self.ds_options["split"]))
        
        error_martinis = glob("/scratch/sx801/scripts/Mol3DGenerator/scripts/martini_shaken/martini_checker/af_frag_2/single_csvs/*..csv")
        error_martinis = set([osp.basename(f).split("..csv")[0] for f in error_martinis])
        
        for split in self.result_mapper.keys():
            this_res = self.result_mapper[split]
            this_save = osp.join(self.save_root, f"{split}_analysis")
            os.makedirs(this_save, exist_ok=True)
            
            pred_errors = (this_res["PROP_TGT"][:, 0] - this_res["PROP_PRED"][:, 0]).abs()
            ax = sns.histplot(pred_errors)
            ax.set(xlabel="Prediction MAE")
            plt.savefig(osp.join(this_save, "prediction_mae"))
            plt.close()

            sorted_index = torch.argsort(pred_errors, descending=True)
            all_info = []
            for i in range(100):
                tested_file = file_locator[split_index[f"{split}_index"][sorted_index[i]]]
                tested_id = osp.basename(tested_file).split(".pth")[0]
                this_info = [pred_errors[sorted_index[i]].item(), tested_id, tested_id in error_martinis]
                all_info.append(this_info)
            with open(osp.join(this_save, "largest_errors.json"), "w") as f:
                json.dump(all_info, f, indent=2)

def main():
    analyzer = TestResultAnalyzer("exp_pl_080_run_2022-10-26_142146__050010", 
                                "exp_pl_080_test_2022-10-27_134305", "..")
    analyzer.run()

if __name__ == "__main__":
    main()
