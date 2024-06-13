from glob import glob
from typing import Optional
from geometry_processors.pl_dataset.lit_pcba_reader import TARGETS
from utils.job_submit.regular_job_submitters import TestJobSubmitter
from utils.utils_functions import lazy_property


import copy
import os
import os.path as osp
import subprocess


class LIT_PCBA_JobSubmitter(TestJobSubmitter):
    def __init__(self, run_dir, debug, ref, wait, target: Optional[str], diffdock_nmdn: bool) -> None:
        super().__init__(run_dir, debug, ref, wait)
        self.run_dir: str = run_dir
        self.wait: bool = wait
        self.target: str = target if target else "All"
        self.diffdock_nmdn: bool = diffdock_nmdn

    @lazy_property
    def job_template(self):
        template_file = "bash_scripts/TMPL_subjob-pl-lit-pcba.sbatch"
        if self.diffdock_nmdn:
            template_file = "bash_scripts/TMPL_subjob-pl-lit-pcba-diffdock-nmdn.sbatch"
        with open(template_file) as f:
            res = f.read()
        return res

    @lazy_property
    def info4tmpl(self):
        info = {}
        info["test_folder"] = self.run_dir
        info["job_name"] = osp.basename(self.folder_reader.cfg["folder_prefix"])
        if self.target != "All":
            ds_name = self.ds_args["test_name"]
            tgt_str = self.target.lower()
            tgt_str += "-diffdock"
            info["ds_config"] = f"configs/test_set_lit-pcba-{tgt_str}_{ds_name}.txt"
            assert osp.exists(info["ds_config"]), info["ds_config"]
        info["target"] = self.target + "-diffdock"
        info["net_id"] = os.environ["USER"]
        info["pcba_extra"] = ""
        return info

    @property
    def sbatch_str(self):
        if self.target != "All":
            return super().sbatch_str

        if self._sbatch_str is not None:
            return self._sbatch_str

        ALL_TARGETS = TARGETS
        model_header = self.job_template.split("##BEGIN_MODEL_SCRIPT")[0]
        model_script = self.job_template.split("##BEGIN_MODEL_SCRIPT")[-1].split("##END_MODEL_SCRIPT")[0]
        score_header = self.job_template.split("##END_MODEL_SCRIPT")[-1].split("##BEGIN_SCORE_SCRIPT")[0]
        score_script = self.job_template.split("##BEGIN_SCORE_SCRIPT")[-1]
        res = model_header.format(**self.info4tmpl)
        # write model testing script (computing on GPU)
        for target in ALL_TARGETS:
            target = target + "-diffdock"
            this_info = copy.copy(self.info4tmpl)
            this_info["target"] = target
            ds_name = self.ds_args["test_name"]
            this_info["ds_config"] = f"configs/test_set_lit-pcba-{target.lower()}_{ds_name}.txt"
            assert osp.exists(this_info["ds_config"]), this_info["ds_config"]
            if self.diffdock_nmdn:
                nmdn_result_folder = glob(f"/scratch/sx801/scripts/DiffDock-NMDN/exp_pl_534_run_2024-01-22_211045__480688/exp_pl_534_test_on_{target.lower()}_*")[0]
                this_info["diffdock_nmdn_result"] = nmdn_result_folder
            this_info["pcba_extra"] = f"--linf9_csv '/scratch/sx801/scripts/DiffDock-NMDN/scripts/benchmark-linf9-xgb/preds/{target.split('-')[0]}/*.csv' "
            res += model_script.format(**this_info)
        res += score_header.format(**self.info4tmpl)
        # write screening score calculation script (CPU only)
        for target in ALL_TARGETS:
            target = target + "-diffdock"
            this_info = copy.copy(self.info4tmpl)
            this_info["target"] = target
            res += score_script.format(**this_info)
        self._sbatch_str = res
        return self._sbatch_str

    def run(self):
        name = osp.basename(self.folder_reader.cfg['folder_prefix'])
        file_handle = f"{self.target}-{name}"
        job_fn = f"SUBMIT_GPU_LIT-PCBA-{file_handle}.sbatch"
        job_score_fn = f"SUBMIT_CPU_LIT-PCBA-SCREEN-{self.target}-{name}.sbatch"

        SPLIT_STR = "#!/bin/bash"
        job_str_list = [SPLIT_STR+s for s in self.sbatch_str.split(SPLIT_STR)[1:]]
        assert len(job_str_list) == 2, job_str_list

        with open(osp.join(self.run_dir, job_fn), "w") as f:
            f.write(job_str_list[0])
        with open(osp.join(self.run_dir, job_score_fn), "w") as f:
            f.write(job_str_list[1])

        run_model_job = osp.join(self.run_dir, job_fn)
        run_score_job = osp.join(self.run_dir, job_score_fn)
        if self.debug:
            print(f">>>> {run_model_job}")
            print(f">>>> {run_score_job}")
            print("-------------")
            return
        sub_job_model = f"sbatch {run_model_job}"
        if self.wait:
            # wait for the current trainning job finishes
            exp_id = osp.basename(self.run_dir.split("_run_")[0])
            current_job = subprocess.run(f"sacct -s r | grep '{exp_id}'", shell=True, check=True, capture_output=True, text=True)
            job_id = current_job.stdout.split()[0]
            sub_job_model = f"sbatch --dependency=afterany:{job_id} {run_model_job}"
            print(f"Test job {run_model_job} will run after job {job_id} finishes.")
        model_job_out = subprocess.run(sub_job_model, shell=True, check=True, capture_output=True, text=True)
        job_id = int(model_job_out.stdout.split()[-1])
        sub_job_casf = f"sbatch --dependency=afterok:{job_id} {run_score_job}"
        print(f"CASF job {run_score_job} will run after job {job_id} finishes.")
        subprocess.run(sub_job_casf, shell=True, check=True)