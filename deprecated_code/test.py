import time
import numpy as np
import torch

from torch_geometric.utils import scatter_

from dataProviders.qm9InMemoryDataset import Qm9InMemoryDataset
from utils.utils_functions import load_data_from_index


def simulate_scatter_(x, batch):
    return scatter_('add', x, batch)


if __name__ == '__main__':
    _device = torch.device('cuda')
    shuffle_index = np.load('../data/split.npz')
    train_index, val_index, test_index = torch.as_tensor(shuffle_index['train']), \
                                         torch.as_tensor(shuffle_index['validation']), \
                                         torch.as_tensor(shuffle_index['test'])
    data = Qm9InMemoryDataset(root='../data', boundary_factor=2)
    edge_index = load_data_from_index(data, val_index[:8]).atom_edge_index
    batch = edge_index[1, :].to(_device)

    t0 = time.time()
    for i in range(1):
        x = torch.randn(1282, 128, device=_device)
        batch = batch[torch.randperm(batch.shape[0])]
        r = simulate_scatter_(x, batch)
    delta_time = time.time() - t0
    print('Finished')
