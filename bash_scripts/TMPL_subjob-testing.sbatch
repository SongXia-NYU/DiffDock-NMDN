#!/bin/bash
#
#SBATCH --job-name=t-{job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu

module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name {test_folder} "
