#!/bin/bash
#
#SBATCH --job-name=vina-lit-pcba
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=16GB
#SBATCH --array=16
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=sx801@nyu.edu

module purge

TARGET=$1

overlays="--overlay /vast/sx801/geometries/LIT-PCBA-DiffDock/$TARGET-batch1.sqf "

for n in {2..9};
do
batch=/vast/sx801/geometries/LIT-PCBA-DiffDock/$TARGET-batch$n.sqf
if [ -f "$batch" ]; then
overlays="$overlays --overlay $batch "
fi
done

polarh=/vast/sx801/geometries/LIT-PCBA-DiffDock/$TARGET-polarh.sqf
if [ -f "$polarh" ]; then
overlays="$overlays --overlay $polarh "
fi

if [ "$SLURM_ARRAY_TASK_ID" == "" ]; then SLURM_ARRAY_TASK_ID=$2; fi

PYTHONPATH="."
PYTHONPATH="/scratch/sx801/scripts/DiffDock:$PYTHONPATH"
PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"

singularity exec  \
    --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
    $overlays \
    --overlay /home/sx801/softwares.sqf \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; export PYTHONPATH=$PYTHONPATH; \
    python lit_pcba.py --target $TARGET --array_id $SLURM_ARRAY_TASK_ID "
