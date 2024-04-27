#!/bin/bash
#
#SBATCH --job-name=t-{job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user={net_id}@nyu.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT

module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name {test_folder}  \
            --explicit_ds_config configs/test_set_deep_accnet_test.txt \
            --only_predict --use_exist --use_tqdm --compute_external_mdn "

/share/apps/singularity/bin/singularity exec --nv \
             --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python scripts/pp_raw2csv.py \
            --folder_name {test_folder} "

exit

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name {test_folder}  \
            --explicit_ds_config configs/test_set_pp_test_af22_50_rankeds.txt \
            --only_predict --use_exist --use_tqdm --compute_external_mdn "

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name {test_folder}  \
            --explicit_ds_config configs/test_set_pp_test_decoys2_separated.txt \
            --only_predict --use_exist --use_tqdm --compute_external_mdn "

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /home/sx801/conda_envs/pth1.12-cuda116-25GB-500K.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python scripts/batch_cal_ppep_scores.py --folder_name {test_folder} "
