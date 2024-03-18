source /ext3/env.sh

export PYTHONPATH=../dataProviders:$PYTHONPATH

python select_conformer.py --pyg_name '$1' --save_folder $2 --pretrained_model $3