#!/bin/bash
#
#SBATCH --job-name=genscore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu

module purge

PYTHONPATH="."
PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"

singularity exec  \
    --overlay /scratch/sx801/singularity-envs/RTMscore-cuda116-25GB-500K.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /home/sx801/softwares.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH; \
    python casf-scoring.py "