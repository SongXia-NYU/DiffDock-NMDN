#!/bin/bash
#
#SBATCH --job-name=mmff_opt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=6GB
#SBATCH --array=0-19
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge


if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then SLURM_ARRAY_TASK_ID=$1; fi

PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN"

singularity exec  \
    --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
    --overlay /home/sx801/softwares.sqf \
    --overlay /vast/sx801/geometries/PDBBind_DiffDock.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH; \
    python pdbbind2020_diffdock_nmdn_mmffrmsd.py --array_id $SLURM_ARRAY_TASK_ID "