from datetime import datetime
import torch
import os
import os.path as osp
import json

from dig.threedgraph.dataset import QM93D
from dig.threedgraph.method import ComENet
from dig.threedgraph.evaluation import ThreeDEvaluator
from dig.threedgraph.method import run

from utils.utils_functions import get_device

# Load the dataset and split
dataset = QM93D(root='/scratch/sx801/temp/dataset/')
target = 'U0'
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Define model, loss, and evaluation
model_args = {"middle_channels": 256, "cutoff": 5.0, "num_layers": 8, "num_radial": 6, "num_spherical": 3}
model = ComENet(**model_args)                 
loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()

# Train and evaluate
current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S__%f')
run3d = run()
device = get_device()
log_dir="exp_comenet_004_run_"+current_time
os.makedirs(log_dir, exist_ok=True)
with open(osp.join(log_dir, "model_args.json"), "w") as f:
    json.dump(model_args, f, indent=2)

train_args = {"epochs": 250, "batch_size": 256, "vt_batch_size": 256, "lr": 0.0002, "lr_decay_factor": 0.5,
              "lr_decay_step_size": 30}
with open(osp.join(log_dir, "train_args.json"), "w") as f:
    json.dump(train_args, f, indent=2)
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, **train_args, log_dir=log_dir)
