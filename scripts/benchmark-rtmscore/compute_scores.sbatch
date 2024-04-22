#!/bin/bash
#
#SBATCH --job-name=CASF
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=/scratch/sx801/scripts/DiffDock-NMDN; \
            python compute_scores.py "
