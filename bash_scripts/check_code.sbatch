#!/bin/bash
#
#SBATCH --job-name=t-exp_pl_250
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:15:00
#SBATCH --mem=12GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=songxia23@gmail.com

module purge

rm -r exp_pl_250_run_2023-03-20_174613__564272/exp_pl_250_test_on_casf2016-*
rm -r exp_pl_250_run_2023-03-20_174613__564272/casf-scores

singularity exec --nv \
            --overlay /home/sx801/conda_envs/pth1.12-cuda116-25GB-500K.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; python test.py --folder_names 'exp_pl_250_run_2023-03-20_174613__564272/' --explicit_ds_config configs/test_set_casf-scoring_dry_atom_pad.txt --use_exist --use_tqdm "

singularity exec --nv \
            --overlay /home/sx801/conda_envs/pth1.12-cuda116-25GB-500K.sqf \
            --overlay /scratch/sx801/data/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; python test.py --folder_names 'exp_pl_250_run_2023-03-20_174613__564272/' --explicit_ds_config configs/test_set_casf-docking_dry_atom_pad.txt --only_predict --use_exist --use_tqdm "

singularity exec --nv \
            --overlay /home/sx801/conda_envs/pth1.12-cuda116-25GB-500K.sqf \
            --overlay /scratch/sx801/data/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=.; python scripts/batch_cal_casf_scores.py --folder_names 'exp_pl_250_run_2023-03-20_174613__564272/'  "
