#!/bin/bash
#
#SBATCH --job-name=ds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu
#SBATCH --array=0-29

module purge

singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.sqf \
            --overlay /vast/sx801/geometries/PDBBind2020_OG.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator:/scratch/sx801/scripts/ESM-GearNet; \
            python scripts/esm_gearnet/proc_pdbbind_array.py --array_id $SLURM_ARRAY_TASK_ID $1 "