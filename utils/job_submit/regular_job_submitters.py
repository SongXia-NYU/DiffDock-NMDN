
import subprocess
import os
import os.path as osp
from typing import Dict

from utils.configs import read_config_file
from utils.train.trainer import data_provider_solver
from utils.eval.trained_folder import TrainedFolder
from utils.utils_functions import lazy_property

class JobSubmitter:
    def __init__(self, debug) -> None:
        self.debug = debug

    def run(self):
        raise NotImplementedError

class TemplateJobSubmitter(JobSubmitter):
    def __init__(self, debug) -> None:
        super().__init__(debug)

        self._sbatch_str = None
        self._job_template = None
        self._info4tmpl = None

    @lazy_property
    def sbatch_str(self) -> str:
        if "net_id" not in self.info4tmpl:
            self.info4tmpl["net_id"] = os.environ["USER"]
        return self.job_template.format(**self.info4tmpl)

    @property
    def job_template(self) -> str:
        raise NotImplementedError

    @property
    def info4tmpl(self) -> Dict[str, str]:
        raise NotImplementedError

class TrainJobSubmitter(TemplateJobSubmitter):
    def __init__(self, config_file, debug) -> None:
        super().__init__(debug)
        self.config_file = config_file

        self._args = None
        self._update_bn_needed = None

    def run(self):
        out_job_file = f"SUBMIT_TRAIN-{osp.basename(self.args['folder_prefix'])}.sbatch"
        with open(out_job_file, "w") as f:
            f.write(self.sbatch_str)

        if not self.debug:
            # production runs should not be in debug mode
            assert not self.args["debug_mode"]
            sub_job_smd = f"sbatch {out_job_file}"
            subprocess.run(sub_job_smd, shell=True, check=True)
        else:
            print(f"-----{self.config_file}-----")
            print(self.sbatch_str)
            print("-------------")

    @property
    def update_bn_needed(self):
        if self._update_bn_needed is None:
            self._update_bn_needed = self.args["loss_metric"] == "mdn" and not self.args["swa_use_buffers"]
        return self._update_bn_needed

    @property
    def info4tmpl(self):
        if self._info4tmpl is None:
            info = {"config_file": self.config_file, "ds_overlay": self.overlay_line(self.args["data_root"], self.args["add_sqf"]),
                    "job_name": osp.basename(self.args["folder_prefix"]), "mem": self.args["mem"]}

            if self.update_bn_needed:
                info["trained_folder"] = f"{self.args['folder_prefix']}*"
            self._info4tmpl = info
        return self._info4tmpl

    @property
    def job_template(self):
        if self._job_template is None:
            templ_file = "bash_scripts/TMPL_subjob.pbs"
            with open(templ_file) as f:
                self._job_template = f.read()

            if self.update_bn_needed:
                with open("bash_scripts/TMPL_subjob_update_bn.pbs") as f:
                    self._job_template += f.read()
        return self._job_template

    @property
    def args(self):
        if self._args is None:
            self._args = read_config_file(self.config_file)
        return self._args


class TestJobSubmitter(TemplateJobSubmitter):
    """
    Running testing jobs.
    """
    def __init__(self, run_dir, debug, ref, wait) -> None:
        super().__init__(debug)
        self.ref = ref
        self.run_dir = run_dir
        # edges are computed on the fly
        self.precompute_edge = False

        self._folder_reader = None
        self._ds_args = None
        self._out_job_file = None

    def run(self):
        with open(osp.join(self.run_dir, self.out_job_file), "w") as f:
            f.write(self.sbatch_str)

        if self.debug:
            print(f"-----{self.run_dir}-----")
            print(self.sbatch_str)
            print("-------------")
            return

        run_model_job = osp.join(self.run_dir, self.out_job_file)
        sub_job_smd = f"sbatch {run_model_job}"
        job_id = self.current_job_id()
        if job_id is not None:
            sub_job_smd = f"sbatch --dependency=afterany:{job_id} {run_model_job}"
            print(f"Test job {run_model_job} will run after job {job_id} finishes.")
        subprocess.run(sub_job_smd, shell=True, check=True)

    def current_job_id(self) -> int:
        exp_id = osp.basename(self.run_dir.split("_run_")[0])
        try:
            current_job = subprocess.run(f"sacct -s r | grep '{exp_id}'", shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            return None
        job_id = current_job.stdout.split()[0]
        return int(job_id)
    
    @lazy_property
    def ds_head(self):
        ds_name = osp.basename(self.ds_args["file_locator"] if "file_locator" in self.ds_args else self.ds_args["dataset_name"])
        for head in ["PL_train-", "PDBind_v2020", "PBind2020", "biolip."]:
            if ds_name.startswith(head): return head

    @lazy_property
    def info4tmpl(self):
        info = {}
        info["test_folder"] = self.run_dir
        info["job_name"] = osp.basename(self.folder_reader.cfg["folder_prefix"])
        return info

    @property
    def out_job_file(self):
        if self._out_job_file is None:
            self._out_job_file = f"SUBMIT_TEST-{osp.basename(self.folder_reader.cfg['folder_prefix'])}.sbatch"
        return self._out_job_file

    @lazy_property
    def job_template(self):
        templ_file = "bash_scripts/TMPL_subjob-testing.sbatch"
        with open(templ_file) as f: return f.read()

    @property
    def ds_args(self):
        if self._ds_args is None:
            __, self._ds_args = data_provider_solver(self.folder_reader.cfg, {})
        return self._ds_args

    @property
    def folder_reader(self):
        if self._folder_reader is None:
            self._folder_reader = TrainedFolder(self.run_dir)
        return self._folder_reader