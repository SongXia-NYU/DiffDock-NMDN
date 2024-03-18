#!/bin/bash
#
#SBATCH --job-name=heavy-gen
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi50:1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=30GB

module purge

singularity exec --nv \
	--overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1-extra-5GB-3.2M.ext3:ro \
            --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1.sqf:ro \
            /scratch/work/public/hudson/images/rocm-4.0.1.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=../dataProviders:$PYTHONPATH; python more_runs/al_heavy_data_gen.py --chk exp376_active_ALL_2021-09-21_170147 "
