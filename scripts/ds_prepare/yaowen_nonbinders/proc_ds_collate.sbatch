#!/bin/bash
#
#SBATCH --job-name=proc_ds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge

PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"
ENV_CMD="export PYTHONPATH=$PYTHONPATH"

RUN_CMD="python $1 proc_ds_crys.py "

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /home/sx801/softwares.sqf \
            --overlay /vast/sx801/geometries/PDBBind2020_OG.sqf \
            --overlay /vast/sx801/geometries/PDBBind_DiffDock.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; $ENV_CMD; $RUN_CMD "
