#!/bin/bash
#
#SBATCH --job-name=t-{job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu
module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /home/sx801/conda_envs/pth1.12-cuda116-25GB-500K.sqf \
            --overlay /scratch/sx801/data/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name {test_folder} \
            --mdn_embed_test "
