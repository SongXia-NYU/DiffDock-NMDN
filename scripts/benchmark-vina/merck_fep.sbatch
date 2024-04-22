#!/bin/bash
#
#SBATCH --job-name=lit-pcba
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

MGLPATH=/softwares/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs
PYTHONPATH=/scratch/sx801/scripts/delta_LinF9_XGB/script:/scratch/sx801/scripts/DiffDock-NMDN

singularity exec  \
    --overlay ~/conda_envs/pth1.10-cuda102-25GB-500K.sqf:ro \
    --overlay /home/sx801/conda_envs/DeltaLinF9XGB-15GB-500K.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/docking.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /home/sx801/softwares.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; conda activate py2.7; export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
        export PATH=/ext3/miniconda3/bin:$PATH; \
        /ext3/miniconda3/bin/python merck_fep.py "