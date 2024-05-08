#!/bin/bash
#
#SBATCH --job-name=step2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB

module purge

## /home/sx801/conda_envs/pymol-pdbfixer-openmm-py38-15GB-500K.sqf
## /home/sx801/conda_envs/py2.6-5GB-3.2M.sqf


PYTHONPATH="/scratch/sx801/scripts/Protein-NMDN"
PYTHONPATH="/scratch/sx801/scripts/DiffDock:$PYTHONPATH"


singularity exec  \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /home/sx801/softwares.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH; \
            python $1 step2_generate_split.py "
