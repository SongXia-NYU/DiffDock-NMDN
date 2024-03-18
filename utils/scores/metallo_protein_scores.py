import os
import os.path as osp
import pandas as pd
from utils.eval.TestedFolderReader import TestedFolderReader

from utils.eval.trained_folder import TrainedFolder
from utils.scores.casf_scores import plot_scatter_info
from utils.utils_functions import lazy_property

METAL_ROOT = "/vast/sx801/geometries/PL_physics_infusion/CY-metalloprotein"


class MetalloProteinScores:
    def __init__(self, folder_name) -> None:
        self.folder_name = folder_name
        self.trained_folder = TrainedFolder(folder_name)
        self.test_reader = TestedFolderReader(osp.basename(folder_name), 
                                              "exp_pl_*_test_on_cyang-metal_*",
                                              osp.dirname(folder_name))

    @lazy_property
    def save_root(self):
        test_dir = osp.join(self.folder_name, "metallo-protein-scores")
        os.makedirs(test_dir, exist_ok=True)
        return test_dir
    
    @staticmethod
    def lig_file2pdb(lig_file: str):
        return osp.basename(lig_file).split("_")[0]

    def run(self):
        scoring_result: dict = self.test_reader.result_mapper["test"]
        sample_id = scoring_result["sample_id"].view(-1).cpu().numpy()
        scoring_df = pd.DataFrame({"sample_id": sample_id, "score": scoring_result["PROP_PRED"].view(-1)})
        record: pd.DataFrame = self.test_reader.only_record()
        scoring_df = scoring_df.join(record.astype({"sample_id": int}).set_index("sample_id"))
        scoring_df["pdb"] = scoring_df["ligand_file"].map(self.lig_file2pdb)
        scoring_df = scoring_df.set_index("pdb")

        tgt_df = pd.read_csv(osp.join(METAL_ROOT, "ci1c00737_si_003.csv"))[["pdb", "pKd"]].set_index("pdb")
        scoring_df = scoring_df.join(tgt_df)

        exp, cal = scoring_df["score"].values, scoring_df["pKd"].values
        r2 = plot_scatter_info(exp, cal, self.save_root, "metalloprotein", "Experimental pKd vs. Calculated pKd")
