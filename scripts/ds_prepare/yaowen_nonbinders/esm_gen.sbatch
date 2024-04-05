#!/bin/bash
#
#SBATCH --job-name=esm
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=55GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge

PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"
ENV_CMD="export PYTHONPATH=$PYTHONPATH"

if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then SLURM_ARRAY_TASK_ID=$1; fi

RUN_CMD="python esm_gen.py "

singularity exec --nv \
            --overlay /scratch/sx801/compress/before/diffdock-15GB-500K.ext3:ro \
            --overlay /home/sx801/softwares.sqf \
            --overlay /vast/sx801/geometries/PDBBind2020_OG.sqf \
            --overlay /vast/sx801/geometries/PDBBind_DiffDock.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; $ENV_CMD; $RUN_CMD "
