#!/bin/bash
#
#SBATCH --job-name=t-{job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=30GB
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu

module purge

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.sqf \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name '{test_folder}' \
            --explicit_ds_config {scoring_config} --only_predict --use_exist --use_tqdm --compute_external_mdn "

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name '{test_folder}' \
            --explicit_ds_config {docking_config} --only_predict --use_exist --use_tqdm --compute_external_mdn "

SCREENING_CONFIG={screening_config}
if true && [[ -f "$SCREENING_CONFIG" ]]; then
  /share/apps/singularity/bin/singularity exec --nv \
              --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
              --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
              /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
              bash -c "source /ext3/env.sh; \
            export PYTHONPATH=.:/scratch/sx801/scripts/Mol3DGenerator; \
            python test.py --folder_name '{test_folder}' \
              --explicit_ds_config {screening_config} --only_predict --use_exist --use_tqdm --compute_external_mdn "
fi

#!/bin/bash
#
#SBATCH --job-name=CASF-{job_name}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:30:00
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user={net_id}@nyu.edu

/share/apps/singularity/bin/singularity exec --nv \
            --overlay /scratch/sx801/singularity-envs/KANO-15GB-500K.ext3:ro \
            --overlay /scratch/sx801/singularity-envs/ocp-py39-50G-10M.ext3:ro \
            --overlay /vast/sx801/geometries/CASF-2016-cyang.sqf \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            bash -c "source /ext3/env.sh; export PYTHONPATH=.; python scripts/batch_cal_casf_scores.py --folder_name '{test_folder}' {casf_extra} "
