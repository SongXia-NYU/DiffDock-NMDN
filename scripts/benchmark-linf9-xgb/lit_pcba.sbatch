#!/bin/bash
#
#SBATCH --job-name=genscore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=6GB
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

MGLPATH=/softwares/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs
PYTHONPATH="/scratch/sx801/scripts/delta_LinF9_XGB/script"
PYTHONPATH="/scratch/sx801/scripts/DiffDock-NMDN:$PYTHONPATH"

singularity exec  \
    --overlay /home/sx801/conda_envs/DeltaLinF9XGB-15GB-500K.sqf \
    --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/docking.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /home/sx801/softwares.sqf \
    $overlays \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
        export PATH=/ext3/miniconda3/envs/ocp-models/bin:$PATH; \
        python lit_pcba.py --target $TARGET --info_gen "

singularity exec  \
    --overlay ~/conda_envs/pth1.10-cuda102-25GB-500K.sqf:ro \
    --overlay /home/sx801/conda_envs/DeltaLinF9XGB-15GB-500K.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/docking.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-BlindDocking/screening.sqf \
    --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
    --overlay /home/sx801/softwares.sqf \
    $overlays \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    bash -c "source /ext3/env.sh; conda activate py2.7; export PYTHONPATH=$MGLPATH:$PYTHONPATH; \
        export PATH=/ext3/miniconda3/bin:$PATH; \
        /ext3/miniconda3/bin/python lit_pcba.py --target $TARGET "