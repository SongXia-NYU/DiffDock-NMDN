import subprocess
from utils.eval.trained_folder import TrainedFolder
from utils.train.trainer import data_provider_solver


import os
import os.path as osp

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
        __, self.ds_args = data_provider_solver(self.folder_reader.cfg, {})

    def generate_jobs(self) -> list:
        info = {}
        ds_name = self.ds_args["test_name"]
        info["test_ds_config"] = f"configs/test_set_{ds_name}.txt"
        info["job_name"] = osp.basename(self.folder_reader.cfg["folder_prefix"])
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