#!/bin/bash
#
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu

module purge

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=/scratch/sx801/scripts/DiffDock-NMDN; \
            python $1 "