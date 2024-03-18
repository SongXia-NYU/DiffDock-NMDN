#!/bin/bash
#
#SBATCH --job-name=kano-ds
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge

RUN_CMD="export PYTHONPATH=.:./kano:/scratch/sx801/scripts/Mol3DGenerator; python scripts/kano/check_kano_ds.py"

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; $RUN_CMD "
