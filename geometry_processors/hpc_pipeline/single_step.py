from typing import Optional
import subprocess

# a single step in the pipeline
class HpcSingleStep:
    def __init__(self, dependency: str = "ok", cwd: str = None, debug_mode: bool=False,
                 name: str = None, sbatch_file: str = None) -> None:
        # "any" or "ok"
        self.dependency = dependency
        # working directory
        self.cwd = cwd
        self.debug_mode = debug_mode
        self.name = name
        self.sbatch_file = sbatch_file
    
    def prepare_sbatch_file(self) -> str:
        assert self.sbatch_file is not None
        return self.sbatch_file
    
    # should return the job_id
    def run(self, prev_job_id: Optional[int]) -> int:
        sbatch_file: str = self.prepare_sbatch_file()
        options = ""
        if prev_job_id is not None: 
            options += f" --dependency=after{self.dependency}:{prev_job_id}"
        if self.name is not None:
            options += f" --job-name={self.name}"
        sub_cmd: str = f"sbatch {options} {sbatch_file}"
        print(">>> ", sub_cmd)
        
        if self.debug_mode: return 0

        # Example job_out: "Submitted batch job 43435874"
        job_out = subprocess.run(sub_cmd, shell=True, check=True, capture_output=True, text=True, cwd=self.cwd)
        prev_job_id = int(job_out.stdout.split()[-1])
        return prev_job_id
    