#!/bin/bash
#
#SBATCH --job-name=genscore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=6GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu

module purge


if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then SLURM_ARRAY_TASK_ID=$2; fi

PYTHONPATH="/scratch/sx801/scripts/delta_LinF9_XGB/script"
PYTHONPATH="/scratch/sx801/scripts/DiffDock:$PYTHONPATH"
PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"

singularity exec  \
    --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/docking.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /home/sx801/softwares.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH; \
    python casf_linf9.py  "