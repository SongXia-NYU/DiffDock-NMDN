# Do NOT delete these two unused imports
# They have to be imported before torchdrug for some reason, otherwise they will fail
import torchvision
from ocpmodels.models.equiformer_v2.edge_rot_mat import InitEdgeRotError

from argparse import ArgumentParser
from ast import parse
import copy
import subprocess
import os
import os.path as osp
from glob import glob
from typing import Dict, List, Optional

from utils.train.trainer import data_provider_solver
from utils.eval.trained_folder import TrainedFolder, read_config_file
from utils.utils_functions import lazy_property

class BatchJobSubmitter:
    def __init__(self, cfg: dict) -> None:
        targets = cfg["targets"]
        factory = JobSubmitterFactory(cfg)
        self.job_submitters: List[JobSubmitter] = [factory.get_submitter(i) for i in targets]

    def run(self):
        for submitter in self.job_submitters: submitter.run()

class JobSubmitter:
    def __init__(self, debug) -> None:
        self._sbatch_str = None
        self.debug = debug

        self._job_template = None
        self._info4tmpl = None

    def run(self):
        raise NotImplementedError

    @lazy_property
    def sbatch_str(self) -> str:
        return self.job_template.format(**self.info4tmpl, net_id=os.environ["USER"])

    @property
    def job_template(self) -> str:
        raise NotImplementedError

    @property
    def info4tmpl(self) -> Dict[str, str]:
        raise NotImplementedError

    @staticmethod
    def overlay_parser(data_root, overlay_list):
        overlay = None
        all_overlays = ["/scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro"]
        if overlay_list is not None:
            all_overlays.extend([osp.join(data_root, sqf) for sqf in overlay_list])
        for sqf in all_overlays:
            if overlay is None:
                overlay = sqf
            else:
                overlay += f" --overlay {sqf}"
            assert osp.exists(sqf.split(":")[0]), sqf
        return overlay.replace("$", "\$")


class JobSubmitterFactory:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def get_submitter(self, tgt: str) -> JobSubmitter:
        cfg = self.cfg
        debug = cfg["debug"]
        wait = cfg["wait"]
        if not osp.isdir(tgt):
            # Training job from config file
            assert osp.basename(tgt).startswith("config-")
            return TrainJobSubmitter(tgt, debug)
        
        # Testing job from a trained directory
        if cfg["lit_pcba"] or cfg["lit_pcba_diffdock"]:
            return LIT_PCBA_JobSubmitter(tgt, debug, cfg["ref"], wait, 
                        cfg["target"], cfg["lit_pcba_diffdock"])
        elif cfg["casf_diffdock"]:
            return CASF_BlindDockJobSubmitter(tgt, debug, cfg["ref"], wait)
        elif osp.basename(tgt.rstrip("/")).startswith(("exp_ppep_", "exp_pp_")):
            return ProtPepTestSubmitter(debug, tgt)
        elif osp.exists(osp.join(tgt, "model.pkl")):
            return MDNEmbedTestSubmitter(tgt, debug)
        else:
            return self.parse_submitter_from_args(tgt)

    def parse_submitter_from_args(self, tgt: str) -> JobSubmitter:
        cfg = self.cfg
        debug = cfg["debug"]
        wait = cfg["wait"]

        ds_args: dict = TestJobSubmitter(tgt, True, False, True).ds_args
        ds_name = osp.basename(ds_args["file_locator"] if "file_locator" in ds_args else ds_args["dataset_name"])
        is_testing_casf = False
        for head in ["PL_train-", "PDBind_v2020", "PBind2020", "biolip."]:
            if ds_name.startswith(head): is_testing_casf = True
        if is_testing_casf:
            return CASF_JobSubmitter(tgt, debug, cfg["ref"], wait)
        return TestJobSubmitter(tgt, debug, cfg["ref"], wait)



class TrainJobSubmitter(JobSubmitter):
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
            info = {"config_file": self.config_file, "ds_overlay": self.overlay_parser(self.args["data_root"], self.args["add_sqf"]),
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


class TestJobSubmitter(JobSubmitter):
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
        info["ds_overlay"] = self.overlay_parser(self.folder_reader.args["data_root"], self.folder_reader.args["add_sqf"])
        info["job_name"] = osp.basename(self.folder_reader.args["folder_prefix"])
        return info

    @property
    def out_job_file(self):
        if self._out_job_file is None:
            self._out_job_file = f"SUBMIT_TEST-{osp.basename(self.folder_reader.args['folder_prefix'])}.sbatch"
        return self._out_job_file

    @lazy_property
    def job_template(self):
        templ_file = "bash_scripts/TMPL_subjob-testing.sbatch"
        with open(templ_file) as f: return f.read()

    @property
    def ds_args(self):
        if self._ds_args is None:
            __, self._ds_args = data_provider_solver(self.folder_reader.args, {})
        return self._ds_args

    @property
    def folder_reader(self):
        if self._folder_reader is None:
            self._folder_reader = TrainedFolder(self.run_dir)
        return self._folder_reader


class CASF_JobSubmitter(TestJobSubmitter):
    def __init__(self, run_dir, debug, ref, wait) -> None:
        super().__init__(run_dir, debug, ref, wait)
        self.model_job_tmpl: str = "SUBMIT_CASF-{job_name}.sbatch"
        self.score_job_tmpl: str = "SUBMIT_CASF-Score-{job_name}.sbatch"

    def run(self):
        SPLIT_STR = "#!/bin/bash"
        job_str_list = [SPLIT_STR+s for s in self.sbatch_str.split(SPLIT_STR)[1:]]
        assert len(job_str_list) == 2, job_str_list

        job_name = f"{osp.basename(self.folder_reader.args['folder_prefix'])}"
        if self.ref:
            job_name += "-ref"
        model_job_sbatch: str = osp.join(self.run_dir, self.model_job_tmpl.format(job_name=job_name))
        score_job_sbatch: str = osp.join(self.run_dir, self.score_job_tmpl.format(job_name=job_name))
        with open(model_job_sbatch, "w") as f:
            f.write(job_str_list[0])
        with open(score_job_sbatch, "w") as f:
            f.write(job_str_list[1])

        if self.debug:
            print(f"-----{self.run_dir}-----")
            print(self.sbatch_str)
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
        info["ds_overlay"] = self.overlay_parser(self.folder_reader.args["data_root"], self.folder_reader.args["add_sqf"])
        info["job_name"] = osp.basename(self.folder_reader.args["folder_prefix"])
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
        docking_config = f"configs/test_set_casf-blind-docking_{ds_name}.txt"
        assert osp.exists(docking_config), docking_config
        screening_config = f"configs/test_set_casf-blind-screening_{ds_name}.txt"
        info["docking_config"] = docking_config
        info["screening_config"] = screening_config
        info["casf_extra"] += "--blind-dock "
        info["casf_extra"] += f"--docking_config {docking_config}"
        return info

class LIT_PCBA_JobSubmitter(TestJobSubmitter):
    def __init__(self, run_dir, debug, ref, wait, target: Optional[str], diffdock: bool) -> None:
        super().__init__(run_dir, debug, ref, wait)
        self.run_dir = run_dir
        self.wait = wait
        target = target + "-diffdock" if target and diffdock else target
        self.target = target
        self.diffdock = diffdock

    @property
    def job_template(self):
        if self._job_template is not None:
            return self._job_template
        
        with open("bash_scripts/TMPL_subjob-pl-lit-pcba.sbatch") as f:
            res = f.read()
            self._job_template = res
        return self._job_template

    @lazy_property
    def info4tmpl(self):
        info = {}
        info["test_folder"] = self.run_dir
        info["ds_overlay"] = self.overlay_parser(self.folder_reader.args["data_root"], self.folder_reader.args["add_sqf"])
        info["job_name"] = osp.basename(self.folder_reader.args["folder_prefix"])
        if self.target is not None:
            ds_name = self.ds_args["test_name"]
            info["ds_config"] = f"configs/test_set_lit-pcba-{self.target.lower()}_{ds_name}.txt"
            assert osp.exists(info["ds_config"]), info["ds_config"] 
        info["target"] = self.target
        return info
    
    @property
    def sbatch_str(self):
        if self.target is not None:
            return super().sbatch_str
        
        if self._sbatch_str is not None:
            return self._sbatch_str
        
        ALL_TARGETS = ["ESR1_ant", "GBA"]
        if self.diffdock: ALL_TARGETS = ["ALDH1", "FEN1", "GBA"]
        model_header = self.job_template.split("##BEGIN_MODEL_SCRIPT")[0]
        model_script = self.job_template.split("##BEGIN_MODEL_SCRIPT")[-1].split("##END_MODEL_SCRIPT")[0]
        score_header = self.job_template.split("##END_MODEL_SCRIPT")[-1].split("##BEGIN_SCORE_SCRIPT")[0]
        score_script = self.job_template.split("##BEGIN_SCORE_SCRIPT")[-1]
        res = model_header.format(**self.info4tmpl)
        # write model testing script (computing on GPU)
        for target in ALL_TARGETS:
            target = target + "-diffdock" if self.diffdock else target
            this_info = copy.copy(self.info4tmpl)
            this_info["target"] = target
            ds_name = self.ds_args["test_name"]
            this_info["ds_config"] = f"configs/test_set_lit-pcba-{target.lower()}_{ds_name}.txt"
            assert osp.exists(this_info["ds_config"]), this_info["ds_config"] 
            res += model_script.format(**this_info)
        res += score_header.format(**self.info4tmpl)
        # write screening score calculation script (CPU only)
        for target in ALL_TARGETS:
            target = target + "-diffdock" if self.diffdock else target
            this_info = copy.copy(self.info4tmpl)
            this_info["target"] = target
            res += score_script.format(**this_info)
        self._sbatch_str = res
        return self._sbatch_str

    def run(self):
        name = osp.basename(self.folder_reader.args['folder_prefix'])
        file_handle = f"{self.target}-{name}" if self.target else f"ALL-{name}"
        job_fn = f"SUBMIT_LIT-PCBA-{file_handle}.sbatch"
        if self.diffdock: job_fn = f"SUBMIT_LIT-PCBA-DiffDock-{file_handle}.sbatch"
        job_score_fn = f"SUBMIT_LIT-PCBA-SCREEN-{self.target}-{name}.sbatch"
        if self.diffdock: job_score_fn = f"SUBMIT_LIT-PCBA-DiffDock-SCREEN-{self.target}-{name}.sbatch"

        SPLIT_STR = "#!/bin/bash"
        job_str_list = [SPLIT_STR+s for s in self.sbatch_str.split(SPLIT_STR)[1:]]
        assert len(job_str_list) == 2, job_str_list

        with open(osp.join(self.run_dir, job_fn), "w") as f:
            f.write(job_str_list[0])
        with open(osp.join(self.run_dir, job_score_fn), "w") as f:
            f.write(job_str_list[1])

        if self.debug:
            print(f"-----{self.run_dir}-----")
            print(self.sbatch_str)
            print("-------------")
            return

        run_model_job = osp.join(self.run_dir, job_fn)
        run_score_job = osp.join(self.run_dir, job_score_fn)
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


class MDNEmbedTestSubmitter(JobSubmitter):
    def __init__(self, rundir, debug) -> None:
        super().__init__(debug)
        self.rundir = rundir
        self.folder_reader = TrainedFolder(self.rundir)
        self.out_job_file = f"SUBMIT_MDN_TEST-{osp.basename(self.folder_reader.args['folder_prefix'])}.sbatch"

    @property
    def job_template(self) -> str:
        with open("bash_scripts/TMPL_subjob-esm-embed-test.sbatch") as f:
            return f.read()
        
    @property
    def info4tmpl(self) -> Dict[str, str]:
        info = {"job_name": self.folder_reader.args["folder_prefix"],
                "test_folder": self.rundir}
        return info
    
    def run(self):
        with open(osp.join(self.rundir, self.out_job_file), "w") as f:
            f.write(self.sbatch_str)

        if not self.debug:
            sub_job_smd = f"sbatch {osp.join(self.rundir, self.out_job_file)}"
            subprocess.run(sub_job_smd, shell=True, check=True)
        else:
            print(f"-----{self.rundir}-----")
            print(self.sbatch_str)
            print("-------------")


class SequentialJobSubmitter:
    def __init__(self, debug) -> None:
        self.debug = debug
    
    def generate_jobs(self) -> list:
        # generate sbatch files and return them
        raise NotImplementedError
    
    def get_init_dependency(self):
        # get the dependent job for the test script
        # for example: you want the test job only run after the completion of model training
        exp_id = self.get_exp_id()
        if exp_id is None:
            return None
        
        try:
            current_job = subprocess.run(f"sacct -s r | grep '{exp_id}'", shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"The model {exp_id} has finished training. Therefore, test job will run immediately.")
            return None
        job_id = current_job.stdout.split()[0]
        return job_id

    def get_exp_id(self):
        return None
    
    def run(self) -> None:
        job_list = self.generate_jobs()
        prev_job_id = self.get_init_dependency()
        for i, job_file in enumerate(job_list):
            sub_cmd = f"sbatch {job_file}"
            if prev_job_id is not None:
                dependency = f"--dependency=afterany:{prev_job_id}" if i==0 else f"--dependency=afterok:{prev_job_id}"
                print(f"Job {job_file} will run after {prev_job_id}")
                sub_cmd = f"sbatch {dependency} {job_file}"
            if self.debug:
                continue
            job_out = subprocess.run(sub_cmd, shell=True, check=True, capture_output=True, text=True)
            prev_job_id = int(job_out.stdout.split()[-1])


class ProtPepTestSubmitter(SequentialJobSubmitter):
    def __init__(self, debug, rundir) -> None:
        super().__init__(debug)
        self.rundir = rundir

        self.folder_reader = TrainedFolder(rundir)
        __, self.ds_args = data_provider_solver(self.folder_reader.args, {})

    def generate_jobs(self) -> list:
        info = {}
        ds_name = self.ds_args["test_name"]
        info["test_ds_config"] = f"configs/test_set_{ds_name}.txt"
        info["job_name"] = osp.basename(self.folder_reader.args["folder_prefix"])
        info["test_folder"] = self.rundir
        test_gpu_job = osp.join(self.rundir, f"SUBMIT_TEST_{ds_name}.sbatch")
        test_tmpl = "bash_scripts/TMPL_subjob-ppep-testing.sbatch"
        with open(test_tmpl) as f:
            test_tmpl = f.read()
        with open(test_gpu_job, "w") as f:
            f.write(test_tmpl.format(**info, net_id=os.environ["USER"]))
        return [test_gpu_job]
    
    def get_exp_id(self):
        return osp.basename(self.rundir.split("_run_")[0])


def main():
    parser = ArgumentParser()
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ref", action="store_true")
    parser.add_argument("--casf-diffdock", action="store_true")
    parser.add_argument("--lit-pcba", action="store_true")
    parser.add_argument("--lit-pcba-diffdock", action="store_true")
    parser.add_argument("--target", default=None, help="None for all targets.")
    parser.add_argument("--wait", action="store_true", help="For testing jobs only. Submit the test job when the training job completes.")
    cfg = parser.parse_args()
    cfg: dict = vars(cfg)

    submitter = BatchJobSubmitter(cfg)
    submitter.run()


if __name__ == "__main__":
    main()
