export PYTHONPATH=../dataProviders:$PYTHONPATH

python active_learning.py --config_name $1 --fixed_train --fixed_valid --action_n_heavy "" --metric ENSEMBLE