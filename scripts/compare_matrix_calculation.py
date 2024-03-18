import json
from Networks.PairedPropLayers.MDNLayer import compute_euclidean_distances_matrix

import time
import torch

from utils.utils_functions import get_device

def compare():
    res = {}
    for device in ["cpu", "cuda"]:
        inited_tensors = []
        for i in range(20):
            X = torch.rand((8, 3_000, 3)).to(device)
            inited_tensors.append(X)
        tik = time.time()
        for X in inited_tensors:
            __ = compute_euclidean_distances_matrix(X, X, None, None)
        tok = time.time()

        res[device] = tok - tik

    device_name = torch.cuda.get_device_name(get_device())
    
    with open(f"compute_time_{device_name}.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    compare()
