import subprocess
import argparse
from typing import List, Optional

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+")
    args = parser.parse_args()
    jobs: List[str] = args.jobs

    exp_id: str = jobs[0].split(".sbatch")[0].split("-")[-1]

    try:
        current_job = subprocess.run(f"sacct -s r | grep '{exp_id}'", shell=True, check=True, capture_output=True, text=True)
        job_id: str = current_job.stdout.split()[0]
    except subprocess.CalledProcessError as e:
        print(f"The model {exp_id} has finished training. Therefore, test job will run immediately.")
        job_id = None
    
    cmd = gen_sub_job(jobs.pop(0), job_id, "afterany")
    print(">>> ", cmd)
    out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

    for job in jobs:
        job_id: str = out.stdout.split()[-1]
        cmd = gen_sub_job(job, job_id, "afterok")
        print(">>> ", cmd)
        out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

def gen_sub_job(job_file: str, dependency: Optional[str], condition="afterok"):
    if dependency is None:
        return f"sbatch {job_file}"
    
    return f"sbatch --dependency={condition}:{dependency} {job_file}"

if __name__ == "__main__":
    main()
