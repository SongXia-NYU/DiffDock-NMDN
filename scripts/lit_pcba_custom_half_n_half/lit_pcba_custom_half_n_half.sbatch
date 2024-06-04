#!/bin/bash
#
#SBATCH --job-name=screen-lit-pcba-diffdock-nmdn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:40:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu



module purge

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/LIT-PCBA.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=/scratch/sx801/scripts/Protein-NMDN:$PYTHONPATH; \
            python $1 lit_pcba_custom_half_n_half.py "

