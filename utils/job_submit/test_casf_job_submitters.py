
import subprocess
import os.path as osp

from utils.utils_functions import lazy_property
from utils.job_submit.regular_job_submitters import TestJobSubmitter


class CASF_JobSubmitter(TestJobSubmitter):
    def __init__(self, run_dir, debug, ref, wait) -> None:
        super().__init__(run_dir, debug, ref, wait)
        self.model_job_tmpl: str = "SUBMIT_CASF-{job_name}.sbatch"
        self.score_job_tmpl: str = "SUBMIT_CASF-Score-{job_name}.sbatch"

    def run(self):
        SPLIT_STR = "#!/bin/bash"
        job_str_list = [SPLIT_STR+s for s in self.sbatch_str.split(SPLIT_STR)[1:]]
        assert len(job_str_list) == 2, job_str_list

        job_name = f"{osp.basename(self.folder_reader.cfg['folder_prefix'])}"
        if self.ref:
            job_name += "-ref"
        model_job_sbatch: str = osp.join(self.run_dir, self.model_job_tmpl.format(job_name=job_name))
        score_job_sbatch: str = osp.join(self.run_dir, self.score_job_tmpl.format(job_name=job_name))
        with open(model_job_sbatch, "w") as f:
            f.write(job_str_list[0])
        with open(score_job_sbatch, "w") as f:
            f.write(job_str_list[1])

        if self.debug:
            print(f">>> {model_job_sbatch}")
            print(f">>> {score_job_sbatch}")
            print("-------------")
            return

        sub_job_model = f"sbatch {model_job_sbatch}"
        # wait for the current trainning job finishes
        job_id = self.current_job_id()
        if job_id is not None:
            sub_job_model = f"sbatch --dependency=afterany:{job_id} {model_job_sbatch}"
            print(f"Test job {model_job_sbatch} will run after job {job_id} finishes.")
        model_job_out = subprocess.run(sub_job_model, shell=True, check=True, capture_output=True, text=True)
        job_id = int(model_job_out.stdout.split()[-1])
        sub_job_casf = f"sbatch --dependency=afterok:{job_id} {score_job_sbatch}"
        print(f"CASF job {score_job_sbatch} will run after job {job_id} finishes.")
        subprocess.run(sub_job_casf, shell=True, check=True)

    @lazy_property
    def info4tmpl(self):
        locator = osp.basename(self.ds_args["file_locator"] if "file_locator" in self.ds_args else self.ds_args["dataset_name"])
        ds_name = locator.split(".loc.")[0].split(self.ds_head)[-1]
        ds_name = "dry_" + ds_name

        # TODO: they are no longer needed
        overlay_scoring = f"/scratch/sx801/data/casf-scoring-{ds_name}.sqf"
        # assert osp.exists(overlay_scoring), overlay_scoring
        overlay_docking = f"/scratch/sx801/data/casf-docking-{ds_name}.sqf"
        # assert osp.exists(overlay_docking), overlay_docking
        overlay_screening = f"/scratch/sx801/data/casf-screening-{ds_name}.sqf"

        if not self.precompute_edge:
            ds_name = "_".join(ds_name.split("_")[:-1])
        ds_name = ds_name.replace("$", "")
        if "test_name" in self.ds_args:
            ds_name = self.ds_args["test_name"]
        scoring_config = f"configs/test_set_casf-scoring_{ds_name}.txt"
        assert osp.exists(scoring_config), scoring_config
        if self.ref:
            ds_name += "-ref"
        docking_config = f"configs/test_set_casf-docking_{ds_name}.txt"
        assert osp.exists(docking_config), docking_config
        screening_config = f"configs/test_set_casf-screening_{ds_name}.txt"

        casf_extra = ""
        if self.ref:
            casf_extra += "--ref"

        info = {"scoring_config": scoring_config, "overlay_scoring": overlay_scoring.replace("$", "\$"),
                "overlay_docking": overlay_docking.replace("$", "\$"), "overlay_screening": overlay_screening.replace("$", "\$"),
                "docking_config": docking_config, "screening_config": screening_config, "casf_extra": casf_extra}
        info["test_folder"] = self.run_dir
        info["job_name"] = osp.basename(self.folder_reader.cfg["folder_prefix"])
        return info
    
    @lazy_property
    def job_template(self):
        templ_file = "bash_scripts/TMPL_subjob-pl-testing.sbatch"
        with open(templ_file) as f: return f.read()

class CASF_BlindDockJobSubmitter(CASF_JobSubmitter):
    def __init__(self, run_dir, debug, ref, wait) -> None:
        super().__init__(run_dir, debug, ref, wait)
        self.model_job_tmpl: str = "SUBMIT_CASF-Blind-{job_name}.sbatch"
        self.score_job_tmpl: str = "SUBMIT_CASF-Blind-Score-{job_name}.sbatch"

    @lazy_property
    def job_template(self):
        templ_file = "bash_scripts/TMPL_subjob-casf-blind-docking.sbatch"
        with open(templ_file) as f: return f.read()
    
    @lazy_property
    def info4tmpl(self):
        info = super().info4tmpl
        ds_name = self.ds_args["test_name"]
        if self.ref: ds_name += "-ref"
        docking_config = f"configs/test_set_casf-blind-docking_{ds_name}.yaml"
        assert osp.exists(docking_config), docking_config
        screening_config = f"configs/test_set_casf-blind-screening_{ds_name}.yaml"
        info["docking_config"] = docking_config
        info["screening_config"] = screening_config
        info["docking_extra"] = ""
        info["screening_extra"] = ""
        if self.folder_reader.cfg.data["diffdock_nmdn_result"] is not None:
            info["docking_extra"] = "--diffdock_nmdn_result /scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/exp_pl_534_test_on_casf2016-blind-docking_2024-03-31_174131 "
            info["screening_extra"] = "--diffdock_nmdn_result /scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/exp_pl_534_test_on_casf2016-blind-screening_2024-03-27_005509"
        # info["casf_extra"] += f"--docking_config {docking_config}"
        return info