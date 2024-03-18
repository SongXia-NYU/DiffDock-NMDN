from utils.eval.TestedFolderReader import TestedFolderReader
import pandas as pd
import os.path as osp

from utils.scores.casf_scores import _protein_file2pdb, plot_scatter_info


def score_ignore_charge():
    run_folder = "exp_pl_022_run_2022-09-09_200038__064356"
    test_folder = "exp_pl_022_test_on_casf2016-scoring_2022-09-09_230606"
    scoring_reader = TestedFolderReader(run_folder, test_folder, "../results")
    scoring_result = scoring_reader.result_mapper["test"]
    record = scoring_reader.only_record()
    scoring_df = pd.DataFrame({"sample_id": scoring_result["sample_id"].view(-1).cpu().numpy(),
                                "score": scoring_result["PROP_PRED"].view(-1).cpu().numpy(),
                                "exp": scoring_result["PROP_TGT"].view(-1).cpu().numpy()})
    scoring_df = scoring_df.join(record.astype({"sample_id": int}).set_index("sample_id"))
    scoring_df["pdb"] = scoring_df["protein_file"].map(_protein_file2pdb)
    scoring_df["#code"] = scoring_df["ligand_file"].map(lambda s: s.split(".sdf")[0])
    charge_df = pd.read_csv("/scratch/sx801/scripts/Mol3DGenerator/scripts/casf-2016/casf_charge.csv")
    scoring_df = scoring_df.set_index("pdb")
    scoring_df = scoring_df.join(charge_df.set_index("pdb"), on="pdb")
    # print(scoring_df)
    mask = (scoring_df["protein_charge"]+scoring_df["ligand_charge"])==0

    cal = scoring_df["score"].values[mask]
    exp = scoring_df["exp"].values[mask]
    r2 = plot_scatter_info(exp, cal, f"../results/{run_folder}/casf-scores", "exp_vs_cal_neutral.png", "Experimental pKd vs. Calculated pKd Neutral only")

if __name__ == "__main__":
    score_ignore_charge()
