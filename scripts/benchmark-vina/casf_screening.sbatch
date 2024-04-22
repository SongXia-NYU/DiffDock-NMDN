#!/bin/bash
#
#SBATCH --job-name=genscore-casf-screen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge

if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then SLURM_ARRAY_TASK_ID=$1; fi

MGLPATH=/softwares/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs
PYTHONPATH="."
PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"

singularity exec  \
    --overlay /scratch/sx801/scripts/GenScore/genscore-25GB-500K.ext3:ro \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/docking.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /home/sx801/softwares.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; \
    export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
    python casf_screening.py --array_id $SLURM_ARRAY_TASK_ID "
