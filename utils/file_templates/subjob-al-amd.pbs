#!/bin/bash
#
#SBATCH --job-name=amd-e{}
#SBATCH --nodes=1
#SBATCH --gres=gpu:mi50:1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=30GB

module purge

singularity exec --nv \
	--overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1-extra-5GB-3.2M.ext3:ro \
            --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1.sqf:ro \
            /scratch/work/public/hudson/images/rocm-4.0.1.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=../dataProviders:$PYTHONPATH; python active_learning.py --config_name config-exp{}.txt --select NUMBER --candidate BELOW_N_HEAVY --number 10000 --init_size 10000 --action_n_heavy '20 20 20 20 20 20 20 20 20' --init_n_heavy 20 --fixed_valid  "
