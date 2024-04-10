"""
Computing RMSD between Diffdock pose and crystal pose 
"""
from collections import defaultdict
from glob import glob
import os.path as osp
import pandas as pd
from rdkit.Chem.AllChem import SDMolSupplier

from geometry_processors.pl_dataset.merck_fep_reader import MerckFEPReader
from utils.rmsd import symmetry_rmsd_from_mols
from utils.scores.casf_blind_scores import CASFBlindDockScore


reader = MerckFEPReader("/vast/sx801/geometries/fep-benchmark")

def compute_rmsd():
    rmsd_info = defaultdict(lambda: [])
    for target in reader.TARGETS:
        src_ligs = reader.get_ligs_src(target)
        mol_supp = SDMolSupplier(src_ligs, removeHs=True, sanitize=True)
        name_mapper = {}
        for mol in mol_supp:
            name = mol.GetProp("_Name")
            if name == "SHP099-1/Example 7": name = "SHP099-1"
            name_mapper[name] = mol
        diffdock_ligs = glob(osp.join(reader.ds_root, "pose_diffdock", "raw_predicts", f"{target}.*"))
        for diffdock_root in diffdock_ligs:
            name = ".".join(osp.basename(diffdock_root).split(".")[1:])
            ref_mol = name_mapper[name]
            for diffdock_sdf in glob(osp.join(diffdock_root, "rank*_confidence*.sdf")):
                rank = osp.basename(diffdock_sdf).split("_")[0]
                if name.startswith("Example"): name = "".join(name.split(" "))
                fl = f"{target}.{name}.{rank}"
                diffdock_mol = SDMolSupplier(diffdock_sdf)[0]
                rmsd = symmetry_rmsd_from_mols(diffdock_mol, ref_mol, 10)
                rmsd_info["file_handle"].append(fl)
                rmsd_info["rmsd"].append(rmsd)
    rmsd_info = pd.DataFrame(rmsd_info)
    rmsd_info.to_csv("/vast/sx801/geometries/fep-benchmark/pose_diffdock/diffdock_crystal_rmsd.csv", index=False)

def compute_docking_power():
    rmsd_df = pd.read_csv("/vast/sx801/geometries/fep-benchmark/pose_diffdock/diffdock_crystal_rmsd.csv")
    rmsd_df["target"] = rmsd_df["file_handle"].map(lambda s: s.split(".")[0])
    rmsd_df = rmsd_df[rmsd_df["target"] != "eg5_alternativeloop"]
    rmsd_df["rank"] = rmsd_df["file_handle"].map(lambda s: int(s.split("rank")[-1]))
    rmsd_df = rmsd_df.set_index("file_handle")
    
    from utils.eval.TestedFolderReader import TestedFolderReader
    scoring_reader = TestedFolderReader("/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/exp_pl_534_test_on_merck_fep-diffdock_2024-02-29_182750")
    scoring_result = scoring_reader.result_mapper["test"]
    res_info = {"sample_id": scoring_result["sample_id"],
                "pKd_score": scoring_result["PROP_PRED"].view(-1).numpy(),
                "NMDN_score": scoring_result["MDN_LOGSUM_DIST2_REFDIST2"].view(-1).numpy()}
    res_df: pd.DataFrame = pd.DataFrame(res_info).set_index("sample_id")
    record: pd.DataFrame = scoring_reader.only_record().set_index("sample_id")
    res_df = res_df.join(record).set_index("file_handle")
    
    rmsd_df = rmsd_df.join(res_df, how="inner").rename({"NMDN_score": "score"}, axis=1)
    rmsd_df["pdb_id"] = rmsd_df.index.map(lambda s: ".".join(s.split(".")[:-1]))
    out_df = CASFBlindDockScore.diffdock_detailed_docking_power(rmsd_df)
    out_df.to_csv(osp.join("scripts/ds_prepare/merck_fep", "docking_detailed.csv"))
    out_df.to_excel(osp.join("scripts/ds_prepare/merck_fep", "docking_detailed.xlsx"), float_format="%.2f")


if __name__ == "__main__":
    compute_docking_power()
