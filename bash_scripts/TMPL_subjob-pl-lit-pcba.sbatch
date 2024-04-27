#!/bin/bash
#
#SBATCH --job-name=t-{target}-{job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=18:00:00
#SBATCH --mem=32GB

##BEGIN_MODEL_SCRIPT

module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/LIT-PCBA.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name '{test_folder}' \
            --explicit_ds_config {ds_config} \
            --use_exist --use_tqdm --only_predict --compute_external_mdn "

##END_MODEL_SCRIPT

#!/bin/bash
#
#SBATCH --job-name=screen-{target}-{job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:40:00
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu

##BEGIN_SCORE_SCRIPT

module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/LIT-PCBA.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=.:$PYTHONPATH; \
            python scripts/batch_cal_lit_pcba_scores.py --folder_name '{test_folder}' \
            --target {target} "

##END_SCORE_SCRIPT
