import argparse
from contextlib import contextmanager
import copy
import gc
import logging
import math
import os
import os.path as osp
import random
import re
import signal
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Union
from tqdm import tqdm
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Dataset
from scipy.spatial import KDTree
from torch.optim.swa_utils import AveragedModel
from torch_scatter import scatter

# Constants:

# Coulomb’s constant in eV A and e
from utils.configs import Config

k_e = 14.399645352
# CPU device, part of the functions are calculated faster in cpu
cpu_device = torch.device('cpu')
# Convert Hartree to eV
hartree2ev = 27.2114
# floating type
floating_type = torch.float32

kcal2ev = 1 / 23.06035
# Atomic reference energy at 0K (unit: Hartree)
atom_ref = {1: -0.500273, 6: -37.846772, 7: -54.583861, 8: -75.064579, 9: -99.718730}

matrix_to_index_map = {}

mae_fn = torch.nn.L1Loss(reduction='mean')
mse_fn = torch.nn.MSELoss(reduction='mean')

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device():
    # we use a function to get device for proper distributed training behaviour
    # return torch.device("cpu")
    return _device


def solv_num_workers():
    try:
        n_cpu_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu_avail = None
    n_cpu = os.cpu_count()
    num_workers = n_cpu_avail if n_cpu_avail is not None else n_cpu
    return n_cpu_avail, n_cpu, num_workers


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def gaussian_rbf(D, centers, widths, cutoff, coe=1., return_dict=False, linear=False):
    """
    The rbf expansion of a distance
    Input D: matrix that contains the distance between to atoms
          K: Number of generated distance features
    Output: A matrix containing rbf expanded distances
    """
    D = D.view(-1, 1)
    if linear:
        effect_dist = D / cutoff
    else:
        effect_dist = torch.exp(-D * coe)

    rbf = _cutoff_fn(D, cutoff) * torch.exp(-widths * (effect_dist - centers) ** 2)
    if return_dict:
        return {"rbf": rbf}
    else:
        return rbf


def _get_index_from_matrix(num, previous_num):
    """
    get the edge index compatible with torch_geometric message passing module
    eg: when num = 3, will return:
    [[0, 0, 0, 1, 1, 1, 2, 2, 2]
    [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    :param num:
    :param previous_num: the result will be added previous_num to fit the batch
    :return:
    """
    if num in matrix_to_index_map.keys():
        return matrix_to_index_map[num] + previous_num
    else:
        index = torch.LongTensor(2, num * num).to(get_device())
        index[0, :] = torch.cat([torch.zeros(num, device=get_device()).long().fill_(i) for i in range(num)], dim=0)
        index[1, :] = torch.cat([torch.arange(num, device=get_device()).long() for _ in range(num)], dim=0)
        mask = (index[0, :] != index[1, :])
        matrix_to_index_map[num] = index[:, mask]
        return matrix_to_index_map[num] + previous_num


matrix_modify = {}


def _get_modify_matrix(num):
    """
    get the modify matrix.
    equivalent to -torch.eye(num)
    data will be stored in matrix_modify to save time when next time need it
    :param num:
    :return:
    """
    if num in matrix_modify.keys():
        return matrix_modify[num]
    else:
        matrix = torch.Tensor(num, num).type(floating_type).zero_()
        for i in range(num):
            matrix[i, i] = -1.
        matrix_modify[num] = matrix
        return matrix


batch_pattern = {}


def _get_batch_pattern(batch_size, max_num):
    """
    get the batch pattern, for example, if batch_size=5, max_num=3
    the pattern will be: [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
    new pattern will be stored in batch_pattern dictionary to avoid recalculation
    :return:
    """
    if batch_size in batch_pattern.keys():
        return batch_pattern[batch_size]
    else:
        pattern = [i // max_num for i in range(batch_size * max_num)]
        batch_pattern[batch_size] = pattern
        return pattern


def _cal_dist(d1, d2):
    """
    calculate the Euclidean distance between d1 and d2
    :param d1:
    :param d2:
    :return:
    """
    delta_R = d1 - d2
    return torch.sqrt(torch.sum(torch.mul(delta_R, delta_R))).view(-1, 1).type(floating_type)


def softplus_inverse(x):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    return torch.log(-torch.expm1(-x)) + x


def _cutoff_fn(D, cutoff):
    """
    Private function called by rbf_expansion(D, K=64, cutoff=10)
    """
    x = D / cutoff
    x3 = x ** 3
    x4 = x3 * x
    x5 = x4 * x

    result = 1 - 6 * x5 + 15 * x4 - 10 * x3
    return result


def _chi_ij(R_ij, cutoff):
    """
    Chi(Rij) function which is used to calculate long-range energy
    return 0 when R_ij = -1 (use -1 instead of 0 to prevent nan when backward)
    :return: Chi(Rij)
    """
    return torch.where(R_ij != -1, _cutoff_fn(2 * R_ij, cutoff) / torch.sqrt(torch.mul(R_ij, R_ij) + 1) +
                       (1 - _cutoff_fn(2 * R_ij, cutoff)) / R_ij, torch.zeros_like(R_ij))


def _correct_q(qi, N, atom_to_mol_batch, q_ref):
    """
    calculate corrected partial_q in PhysNet
    :param qi: partial charge predicted by PhysNet, shape(-1, 1)
    :return: corrected partial_q, shape(-1, 1)
    """
    # predicted sum charge
    Q_pred = scatter(reduce='add', src=qi, index=atom_to_mol_batch, dim=0)

    correct_term = (Q_pred - q_ref) / (N.type(floating_type).to(get_device()))
    # broad cast according to batch to make sure dim correct
    broadcasted_correct_term = correct_term.take(atom_to_mol_batch)
    return qi - broadcasted_correct_term


def cal_coulomb_E(qi: torch.Tensor, edge_dist, edge_index, cutoff, q_ref, N, atom_mol_batch):
    """
    Calculate coulomb Energy from chi(Rij) and corrected q
    Calculate ATOM-WISE energy!
    :return: calculated E
    """
    # debug: cal passed time to improve code efficiency
    # print('&' * 40)
    # t0 = time.time()

    cutoff = cutoff.to(get_device())

    # debug: cal passed time to improve code
    # print('T--------pre cal: ', time.time() - t0)
    if q_ref is not None:
        """
        This correction term is from PhysNet paper:
        As neural networks are a purely numerical algorithm, it is not guaranteed a priori
        that the sum of all predicted atomic partial charges qi is equal
        to the total charge Q (although the result is usually very close
        when the neural network is properly trained), so a correction
        scheme like eq 14 is necessary to guarantee charge
        conservation
        """
        assert N is not None
        assert atom_mol_batch is not None
        qi = _correct_q(qi, N, atom_mol_batch, q_ref)

    # Qi will be corrected according to the paper
    # qi = _partial_q(qi, N, atom_to_mol_batch, Q)

    # debug: cal passed time to improve code
    # print('T--------cal partial qi: ', time.time() - t0)

    # debug: cal passed time to improve code
    # print('T--------split: ', time.time() - t0)

    # Probably I should do qi = qi.clone() to avoid inplace calculation
    q_first = qi.take(edge_index[0, :]).view(-1, 1)
    q_second = qi.take(edge_index[1, :]).view(-1, 1)
    revised_dist = _chi_ij(edge_dist, cutoff=cutoff)
    coulomb_E_terms = q_first * revised_dist * q_second
    '''
    set dim_size here in case the last batch has only one 'atom', which will cause dim to be 1 less because no 
    edge will be formed th that way 
    '''
    coulomb_E = scatter(reduce='add', src=coulomb_E_terms.view(-1), index=edge_index[0, :], dim_size=qi.shape[0], dim=0)

    # debug: cal passed time to improve code
    # print('T--------for loop: ', time.time() - t0)
    # print('&' * 40)

    # times 1/2 because of the double counting
    return (coulomb_E / 2).to(get_device())


def cal_p(qi, R, atom_to_mol_batch):
    """
    Calculate pi from qi and molecule coordinate
    :return: pi
    """

    tmp = torch.mul(qi.view(-1, 1), R.to(get_device()))
    p = scatter(reduce='add', src=tmp, index=atom_to_mol_batch.to(get_device()), dim=0)
    return p


def cal_edge(R, N, prev_N, edge_index, cal_coulomb=True):
    """
    calculate edge distance from edge_index;
    if cal_coulomb is True, additional edge will be calculated without any restriction
    :param cal_coulomb:
    :param prev_N:
    :param edge_index:
    :param R:
    :param N:
    :return:
    """
    if cal_coulomb:
        '''
        IMPORTANT: DO NOT use num(tensor) itself as input, which will be regarded as dictionary key in this function,
        use int value(num.item())
        Using tensor as dictionary key will cause unexpected problem, for example, memory leak
        '''
        coulomb_index = torch.cat(
            [_get_index_from_matrix(num.item(), previous_num) for num, previous_num in zip(N, prev_N)], dim=-1)
        points1 = R[coulomb_index[0, :], :]
        points2 = R[coulomb_index[1, :], :]
        coulomb_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
        coulomb_dist = torch.sqrt(coulomb_dist)

    else:
        coulomb_dist = None
        coulomb_index = None

    short_range_index = edge_index
    points1 = R[edge_index[0, :], :]
    points2 = R[edge_index[1, :], :]
    short_range_dist = torch.sum((points1 - points2) ** 2, keepdim=True, dim=-1)
    short_range_dist = torch.sqrt(short_range_dist)
    return coulomb_dist, coulomb_index, short_range_dist, short_range_index


def get_batch(atom_map, max_num):
    """
    from map to batch
    :param atom_map:
    :param atom_map, batch_size, max_num:
    :return: batch, example: [0,0,0,0,0,0,1,1,2,2,2,...]
    """
    batch_size = atom_map.shape[0] // max_num
    pattern = _get_batch_pattern(batch_size, max_num)
    return torch.LongTensor(pattern)[atom_map]


def get_uniform_variance(n1, n2):
    """
    get the uniform variance to initialize the weight of DNNs suggested at
    Glorot,X.;Bengio,Y. Understanding the Difficulty of Training Deep Feed forward Neural Networks.
    Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 2010; pp 249–256.
    :param n1: the size of previous layer
    :param n2: the size of next layer :return: uniform variance
    """
    return math.sqrt(6) / math.sqrt(n1 + n2)


# generates a random square orthogonal matrix of dimension dim
def square_orthogonal_matrix(dim=3, seed=None):
    random_state = np.random
    if seed is not None:  # allows to get the same matrix every time
        random_state.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

# generates a random square orthogonal matrix of dimension dim using PyTorch
def square_orthogonal_matrix_th(dim=3, seed=None):
    device = get_device()
    random_state = torch.random
    if seed is not None:  # allows to get the same matrix every time
        random_state.seed(seed)
    H = torch.eye(dim).to(device)
    D = torch.ones((dim,)).to(device)
    for n in range(1, dim):
        size=(dim - n + 1,)
        x = torch.normal(torch.zeros(size), torch.ones(size)).to(device)
        D[n - 1] = torch.sign(x[0]).to(device)
        x[0] -= D[n - 1] * torch.sqrt((x * x).sum())
        # Householder transformation
        Hx = (torch.eye(dim - n + 1).to(device) - 2. * torch.outer(x, x) / (x * x).sum())
        mat = torch.eye(dim).to(device)
        mat[n - 1:, n - 1:] = Hx
        H = torch.matmul(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(torch.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H.cpu()


# generates a random (semi-)orthogonal matrix of size NxM
def semi_orthogonal_matrix(N, M, seed=None):
    if N > M:  # number of rows is larger than number of columns
        square_matrix = square_orthogonal_matrix_th(dim=N, seed=seed)
    else:  # number of columns is larger than number of rows
        square_matrix = square_orthogonal_matrix_th(dim=M, seed=seed)
    return square_matrix[:N, :M]


# generates a weight matrix with variance according to Glorot initialization
# based on a random (semi-)orthogonal matrix
# neural networks are expected to learn better when features are decorrelated
# (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
# "Dropout: a simple way to prevent neural networks from overfitting",
# "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
def semi_orthogonal_glorot_weights(n_in, n_out, scale=2.0, seed=None):
    W = semi_orthogonal_matrix(n_in, n_out, seed=seed)
    W *= torch.sqrt(scale / ((n_in + n_out) * W.var()))
    return torch.Tensor(W).type(floating_type).t()


def get_atom_to_efgs_batch(efgs_batch, num_efgs, atom_to_mol_mask):
    # efgs_batch: shape(n_batch, 29)
    # num_efgs: shape(n_batch)

    _batch_corrector = torch.zeros_like(num_efgs)
    _batch_corrector[1:] = num_efgs[:-1]
    batch_size = _batch_corrector.shape[0]
    for i in range(1, batch_size):
        _batch_corrector[i] = _batch_corrector[i - 1] + _batch_corrector[i]
    _batch_corrector = _batch_corrector.view(-1, 1)  # make sure correct dimension for broadcasting
    efgs_batch = efgs_batch + _batch_corrector
    atom_to_efgs_batch = efgs_batch.view(-1)[atom_to_mol_mask]
    return atom_to_efgs_batch


def get_kd_tree_array(R, N):
    """
    Used in data_provider, encapsulate coordinates and numbers into a kd_tree array(tensor)
    :param R: Coordinates
    :param N: Number of atoms in this molecule
    :return: tensor of KD_Tree instances
    """
    num_molecules = R.shape[0]
    kd_trees = np.empty(num_molecules, dtype=KDTree)
    for i in range(num_molecules):
        kd_trees[i] = KDTree(R[i, :N[i].item(), :])
    return kd_trees


def atom_mean_std(E, N, index):
    """
    calculate the mean and stand variance of Energy in the training set
    :return:
    """
    mean = 0.0
    std = 0.0
    num_mol = len(index)
    for _i in range(num_mol):
        i = index[_i]
        m_prev = mean
        x = E[i] / N[i]
        mean += (x - mean) / (i + 1)
        std += (x - mean) * (x - m_prev)
    if isinstance(std, torch.Tensor):
        std = torch.sqrt(std / num_mol)
    else:
        std = math.sqrt(std / num_mol)
    return mean, std


def _pre_nums(N, i):
    return N[i - 1] if i > 0 else 0


def _cal_dim(key):
    return -1 if re.search("index", key) else 0


def dime_edge_expansion(R, edge_index, msg_edge_index, n_dime_rbf, dist_calculator, bessel_calculator,
                        feature_interact_dist, cos_theta=True, return_dict=False, **kwargs):
    # t0 = record_data('edge_msg_gen.load_data', t0)

    """
    calculating bonding infos
    those data will be used in DimeNet modules
    """
    dist_atom = dist_calculator(R[edge_index[0, :], :], R[edge_index[1, :], :])
    rbf_ji = bessel_calculator.cal_rbf(dist_atom, feature_interact_dist, n_dime_rbf)

    # t0 = record_data('edge_msg_gen.bond_rbf', t0)

    dist_msg = dist_calculator(
        R[edge_index[0, msg_edge_index[1, :]], :], R[edge_index[1, msg_edge_index[1, :]], :]
    ).view(-1, 1)
    angle_msg = cal_angle(
        R, edge_index[:, msg_edge_index[0, :]], edge_index[:, msg_edge_index[1, :]], cos_theta).view(-1, 1)
    sbf_kji = bessel_calculator.cal_sbf(dist_msg, angle_msg, feature_interact_dist)

    # t0 = record_data('edge_msg_gen.bond_sbf', t0)
    if return_dict:
        return {"rbf_ji": rbf_ji, "sbf_kji": sbf_kji}
    else:
        return rbf_ji, sbf_kji


def get_n_params(model, logger=None, only_trainable=False):
    """
    Calculate num of parameters in the model
    :param only_trainable: Only count trainable
    :param logger:
    :param model:
    :return:
    """
    result = ''
    counted_params = []
    for name, param in model.named_parameters():
        if not (only_trainable and not param.requires_grad):
            if logger is not None:
                logger.info('{}: {}'.format(name, param.data.shape))
            result = result + '{}: {}\n'.format(name, param.data.shape)
            counted_params.append(param)
    return sum([x.nelement() for x in counted_params]), result


def cal_angle(R, edge1, edge2, cal_cos_theta):
    delta_R1 = R[edge1[0, :], :] - R[edge1[1, :], :]
    delta_R2 = R[edge2[0, :], :] - R[edge2[1, :], :]
    inner = torch.sum(delta_R1 * delta_R2, dim=-1)
    delta_R1_l = torch.sqrt(torch.sum(delta_R1 ** 2, dim=-1))
    delta_R2_l = torch.sqrt(torch.sum(delta_R2 ** 2, dim=-1))
    cos_theta = inner / (delta_R1_l * delta_R2_l + 1e-7)
    if cal_cos_theta:
        angle = cos_theta
    else:
        angle = torch.acos(cos_theta)
    return angle.view(-1, 1)


def get_tensors():
    """
    print out tensors in current system to debug memory leak
    :return: set of infos about tensors
    """
    result = {"set_init"}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tup = (obj.__hash__(), obj.size())
                result.add(tup)
        except:
            pass
    print('*' * 30)
    return result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def info_resolver(s):
    """
    Internal function which resolve expansion function into details, eg:
    gaussian_64_10.0 means gaussian expansion, n=64 and cutoff=10.0
    :param s:
    :return:
    """
    # newer implementation for specifying RBFs
    if "[" in s and "]" in s:
        name, options = option_solver(s, type_conversion=True, return_base=True)
        defaults = {}
        if name == "gaussian":
            defaults = {"dens_min": 0., "coe": 1., "linear": False}
        defaults.update(options)
        defaults["name"] = name
        return defaults

    # legacy version of splitting for compatibility
    info = s.split('_')
    result = {'name': info[0]}
    if info[0] == 'bessel' or info[0] == 'gaussian':
        result['n'] = int(info[1])
        result['dist'] = float(info[2])
        if len(info) == 3:
            # backward compatibility
            result["coe"] = 1.
            result["dens_min"] = 0.
            result["linear"] = False
            return result

        result["coe"] = float(info[3])  
    elif info[0] == 'defaultDime':
        result['n'] = int(info[1])
        result['envelop_p'] = int(info[2])
        result['n_srbf'] = int(info[3])
        result['n_shbf'] = int(info[4])
        result['dist'] = float(info[5])
    elif info[0] == 'coulomb':
        result['dist'] = float(info[1])
    elif info[0].lower() == 'none':
        pass
    else:
        raise ValueError(f"Invalid expansion function {s} !")
    return result


def expansion_splitter(s: Optional[str]) -> Dict[str, str]:
    """
    Internal use only
    Strip expansion function into a dictionary
    :param s:
    :return:
    """
    if s is None: return {}
    
    result = {}
    for mapping in s.split(' '):
        value = mapping.split(':')[1]
        keys = mapping.split(':')[0]
        if keys[0] == '(':
            assert keys[-1] == ')'
            keys = keys[1:-1]
            for key in keys.split(','):
                result[key.strip()] = value
        else:
            result[keys.strip()] = value
    return result


def error_message(value, name):
    raise ValueError('Invalid {} : {}'.format(name, value))


def print_val_results(dataset_name, loss, emae, ermse, qmae, qrmse, pmae, prmse):
    log_info = 'Validating {}: '.format(dataset_name)
    log_info += (' loss: {:.6f} '.format(loss))
    log_info += ('emae: {:.6f} '.format(emae))
    log_info += ('ermse: {:.6f} '.format(ermse))
    log_info += ('qmae: {:.6f} '.format(qmae))
    log_info += ('qrmse: {:.6f} '.format(qrmse))
    log_info += ('pmae: {:.6f} '.format(pmae))
    log_info += ('prmse: {:.6f} '.format(prmse))
    return log_info


def option_solver(option_txt, type_conversion=False, return_base=False):
    option_base = option_txt.split('[')[0]
    if len(option_txt.split('[')) == 1:
        result = {}
    else:
        # option_txt should be like :    '[n_read_out=2,other_option=value]'
        # which will be converted into a dictionary: {n_read_out: 2, other_option: value}
        option_txt = option_txt.split('[')[1]
        option_txt = option_txt[:-1]
        result = {argument.split('=')[0].strip(): argument.split('=')[1].strip()
                  for argument in option_txt.split(',')}
        if type_conversion:
            for key in result.keys():
                value_final = copy.copy(result[key])
                try:
                    tmp = float(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                try:
                    tmp = int(value_final)
                    result[key] = tmp
                except ValueError:
                    pass

                if result[key] in ["True", "False"]:
                    result[key] = (result[key] == "True")
    if return_base:
        return option_base, result
    else:
        return result


def init_model_test(cfg: Config, state_dict, ds: Dataset):
    from Networks.PhysDimeNet import PhysDimeNet
    model = PhysDimeNet(cfg=cfg, ds=ds).to(get_device())
    model = AveragedModel(model, use_buffers=cfg.training.swa_use_buffers)
    if "n_averaged" not in state_dict.keys():
        model.module.load_state_dict(state_dict)
    else:
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        for key in incompatible_keys.missing_keys:
            if not key.startswith("module.expansion_coe"):
                raise ValueError(f"Error(s) in loading state_dict: {incompatible_keys}")
    return model


def non_collapsing_folder(folder_prefix: str, identify="_run_"):
    while True:
        current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S__%f')
        run_directory = folder_prefix + identify + current_time
        time.sleep(random.Random().random() * 3)
        if not os.path.exists(run_directory):
            # add randomness to avoid collapsing
            os.makedirs(run_directory, exist_ok=False)
            return run_directory
        else:
            rd_sleep = random.Random().random() * 20
            print(f"Folder exists, trying to wait {rd_sleep} seconds")
            time.sleep(rd_sleep)


def add_parser_arguments(parser: argparse.ArgumentParser):
    """
    add arguments to parser
    :param parser:
    :return: added parser
    """
    #--------------------------------Deep Learning Model Arguments--------------------------------#
    # Physnet layers
    parser.add_argument("--n_atom_embedding", type=int, default=95)
    parser.add_argument('--modules', type=str, help="eg: D P D P D P, D for DimeNet and P for PhysNet")
    parser.add_argument('--bonding_type', type=str, 
        help="eg: B N B N B N, B for bonding-edge, N for non-bonding edge, L for long-range interaction and " + \
        "BN for both bonding and non-bonding")
    parser.add_argument('--n_feature', type=int)
    parser.add_argument('--n_phys_atomic_res', type=int)
    parser.add_argument('--n_phys_interaction_res', type=int)
    parser.add_argument('--n_phys_output_res', type=int)
    parser.add_argument("--n_output", type=int)
    parser.add_argument("--trioMPW", action="store_true", 
        help="use three linear layers when performing message passing for PP, PL, LL")
    parser.add_argument("--trioMPW_zeroW", action="store_true", 
        help="zero the weights for PL and PP, only LL is randomly initialized")
    parser.add_argument("--preserve_prot_embed", action="store_true", 
        help="Zero_ the weights during initialization to preserve protein embedding at first step")
    # MDN layers
    parser.add_argument('--n_mdn_hidden', type=int, default=None)
    parser.add_argument("--n_mdn_lig_metal_hidden", type=int, default=None)
    parser.add_argument("--n_mdnprop_hidden", type=int, default=None)
    parser.add_argument("--n_mdn_layers", type=int, help="Number of MLP layer in the MDN layer", default=1)
    parser.add_argument("--n_mdnprop_layers", type=int, default=1)
    parser.add_argument("--cross_mdn_prop_name", default="pair_prob_transformed", help="pair_prob_transformed | pair_nll_transformed")
    parser.add_argument("--cross_mdn_behaviour", default="pair_mean", help="pair_mean | mol_sum_mean")
    parser.add_argument("--mdn_threshold_train", type=float, default=None)
    parser.add_argument("--mdn_threshold_eval", type=float, default=None)
    parser.add_argument("--mdn_threshold_prop", type=float, default=None)
    parser.add_argument("--mdn_voronoi_edge", action="store_true")
    parser.add_argument("--mdn_dist_expansion", default=None, help="the distance expansion function for MDN paired properties prediction")
    parser.add_argument("--pair_prop_dist_coe", default=None, type=str, help="regularize pair property by pair distance: inverse | inverse_square")
    parser.add_argument("--martini2aa_action", default="replace_with_aa_feats", help="replace_with_aa_feats | ignore")
    parser.add_argument("--n_mdn_gauss", type=int, default=10, help="Number of Gaussians during MDN calculation.")
    parser.add_argument("--pkd_phys_terms", type=str, default=None, help="Physical terms when predicting pKd. See MPNNPairedPropLayer.register_pkd_phys_terms()")
    parser.add_argument("--pkd_phys_concat", action="store_true")
    parser.add_argument("--pkd_phys_norm", type=float, default=None)
    parser.add_argument("--auxprop_nmdn_name", default="MDN_LOGSUM", type=str, help="Used in NMDN_AuxPropLayer")
    parser.add_argument("--auxprop_nmdn_compute_ref", action="store_true")
    parser.add_argument("--protprot_exclude_edge", type=int, default=None, help="Exclude close interactions.")
    parser.add_argument("--n_paired_mdn_readout", type=int, default=1)
    parser.add_argument("--n_paired_mdn_readout_hidden", type=int, default=None)
    parser.add_argument("--metal_atom_embed_path", type=str, default=None)
    parser.add_argument("--metal_atom_embed_slice", type=int, default=None)
    parser.add_argument("--w_lig_metal", default=1.0, type=float)
    # Dimenet layers
    parser.add_argument('--n_dime_before_residual', type=int)
    parser.add_argument('--n_dime_after_residual', type=int)
    parser.add_argument('--n_output_dense', type=int)
    parser.add_argument('--n_bi_linear', type=int)
    # Normalization layers
    parser.add_argument('--normalize', type=str, default="True")
    parser.add_argument('--shared_normalize_param', type=str, default="False")
    parser.add_argument("--uni_task_ss", type=str, default="False", help="Universal scale/shift for all tasks.")
    parser.add_argument("--train_shift", type=str, default="True")
    # KANO
    parser.add_argument("--kano_ckpt", type=str, default=None)
    # ComENet
    parser.add_argument("--comenet_cutoff", default=8.0, type=float)
    parser.add_argument("--comenet_num_layers", default=4, type=int)
    parser.add_argument("--comenet_num_radial", default=3, type=int)
    parser.add_argument("--comenet_num_spherical", default=2, type=int)
    parser.add_argument("--comenet_num_output_layers", default=3, type=int)
    # Equiformer V2
    parser.add_argument("--equiformer_v2_ckpt", type=str, default=None)
    parser.add_argument("--equiformer_v2_for_energy", action="store_true")
    parser.add_argument("--equiformer_v2_narrow_embed", action="store_true")
    # Misc
    parser.add_argument('--activations', type=str, help='swish | ssp')
    parser.add_argument('--expansion_fn', type=str, default=None)
    parser.add_argument('--restrain_non_bond_pred', type=str, default="False")
    parser.add_argument('--uncertainty_modify', type=str, default='none',
        help="none | concreteDropoutModule | concreteDropoutOutput | swag_${start}_${freq}")
    parser.add_argument('--coulomb_charge_correct', type=str, default="False",
        help='calculate charge correction when calculation Coulomb interaction')
    parser.add_argument("--pooling", type=str, default="sum", help="sum | mem_pooling[heads=?,num_clusters=?,tau=?,n_output=?]")
    parser.add_argument("--batch_norm", type=str, default="False")
    parser.add_argument("--dropout", type=str, default="False")
    parser.add_argument("--requires_atom_embedding", type=str, default="False")
    parser.add_argument("--lin_last", type=str, default="False", help="vi: scale, shift -> sum -> lin")
    parser.add_argument("--last_lin_bias", type=str, default="False")
    parser.add_argument("--acsf", type=int, default=None, help="The dimension of ACSF embedding. By default, ACSF is disabled.")
    parser.add_argument("--mask_z", type=str, default="False")
    parser.add_argument("--ext_atom_features", type=str, default=None)
    #----------------------------------------------------------------------------------------------#

    #--------------------------------------Training Arguments--------------------------------------#
    # Training: optimizations
    parser.add_argument('--optimizer', type=str, default='emaAms_0.999', help="emaAms_${ema} | sgd")
    parser.add_argument('--ema_decay', type=float, help='Deprecated, use --optimizer option instead')
    parser.add_argument('--max_norm', type=float)
    parser.add_argument("--error_if_nonfinite", action="store_true", help="for torch.nn.utils.clip_grad_norm_")
    parser.add_argument("--swa_use_buffers", action="store_true")
    parser.add_argument("--swa_start_step", type=int, default=0, help="The step to enable SWA. By default it will be enabled at the beginning.")
    # Training: loss functions
    parser.add_argument("--loss_metric", type=str, default="mae", help="mae|rmse|mse|ce|bce|evidential|mdn|mdn_mae")
    parser.add_argument('--l2lambda', type=float)
    parser.add_argument('--nh_lambda', type=float)
    parser.add_argument('--force_weight', type=float)
    parser.add_argument('--charge_weight', type=float)
    parser.add_argument('--dipole_weight', type=float)
    parser.add_argument("--action", type=str, default="E", 
        help="name of target, must be consistent with name in data_provider, default E is for PhysNet energy")
    parser.add_argument("--target_names", type=str, action="append", default=[],
        help="For Frag20-solvation: gasEnergy | watEnergy | octEnergy | CalcSol | OctSol")
    parser.add_argument("--regression_ignore_nan", action="store_true", help="Ignore NaNs when "
        "computing regression loss. It is used when only part of training examples has the prperty."
        "For example, on BioLip, only 40k PL pairs have experimental pKd.")
    parser.add_argument("--auto_sol", type=str, default="False",
        help="Automatic calculate solvation energy by subtracting solvent energy by gas energy.")
    parser.add_argument("--auto_sol_no_conv", action="store_true", help="do not convert unit")
    parser.add_argument("--target_nodes", type=str, default="False",
        help="Add extra nodes (fake atoms) for each target, the result of each target will be the aggregated repr of each node.")
    parser.add_argument("--w_mdn", default=1., type=float, help="When using mixed MDN layer with regression layer")
    parser.add_argument("--w_regression", default=1., type=float, help="When using mixed MDN layer with regression layer")
    parser.add_argument("--w_cross_mdn_pkd", default=0., type=float, help="aux task of combining MDN and pKd loss.")
    parser.add_argument("--z_loss_weight", type=float, default=0)
    parser.add_argument("--keep", type=str, default=None)
    parser.add_argument("--flex_sol", action="store_true",
        help="Multi-task FT on experimental datasets: use MT when available, otherwise use st")
    parser.add_argument("--ligand_only", action="store_true", help="Only retain ligand atoms")
    parser.add_argument("--lamda_sol", default=None, type=float)
    parser.add_argument("--auto_pl_water_ref", action="store_true")
    parser.add_argument("--wat_ref_file", default=None, help="calculation results for water reference energy.")
    parser.add_argument("--mdn_w_lig_atom_types", type=float, default=0.)
    parser.add_argument("--mdn_w_prot_atom_types", type=float, default=0.)
    parser.add_argument("--mdn_w_lig_atom_props", type=float, default=0.)
    parser.add_argument("--mdn_w_prot_sasa", type=float, default=0.)
    parser.add_argument("--delta_learning_pkd", action="store_true", help="Delta machine learning on pKd prediction, ST")
    parser.add_argument("--mask_atom", action="store_true", help="Only predict part of the atomic properties")
    # Training: learning rate schedule
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--scheduler', type=str, default='StepLR', help="StepLR | ReduceLROnPlateau")
    parser.add_argument('--decay_steps', type=int)
    parser.add_argument('--decay_rate', type=float)
    parser.add_argument('--warm_up_steps', type=int, help="Steps to warm up")
    # Training: finetuning/transfer learning
    parser.add_argument('--use_trained_model', type=str, default="False")
    parser.add_argument("--ft_discard_training_model", action="store_true", 
        help="Use shadow model (best model) to initialize both training and shadow model." + \
        "This is helpful when you want to freeze some parameters without messing up the weights by SWA.")
    parser.add_argument('--freeze_option', type=str, default='none', help='none | prev | prev_extra')
    parser.add_argument('--reset_optimizer', type=str, default="True",
        help='If true, will reset optimizer/scheduler regardless of if you use pretrained model or not')
    parser.add_argument("--reset_output_layers", type=str, default="False")
    parser.add_argument("--reset_scale_shift", type=str, default="False")
    parser.add_argument("--reset_ptn", action="append", type=str, default=[])
    parser.add_argument("--ft_lr_factor", type=float, default=None)
    parser.add_argument("--normal_lr_ptn", action="append", 
        help="parameter names that use normal learning rate. Others are reduced by ft_lr_factor")
    parser.add_argument("--lower_lr_ptn", action="append", 
        help="parameter names that lower normal learning rate. Exclusive with normal_lr_ptn")
    parser.add_argument("--mdn_freeze_bn", action="store_true", help="Freeze the BatchNorm in MDN for proper transfer learning behaviour.")
    # Training: controls
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--test_interval', type=str, help="DONT USE! For compatibility only, no longer used.")
    parser.add_argument('--early_stop', type=int, default=-1, help="early stopping, set to -1 to disable")
    parser.add_argument("--stop_low_lr", action="store_true")
    # Training: muti-gpu
    parser.add_argument("--local_rank", type=int, default=0)
    # Validation
    parser.add_argument("--val_pair_prob_dist_coe", default=None, type=str, help="regularize pair MDN prob by pair distance: inverse | inverse_square")
    parser.add_argument("--hist_pp_intra_mdn", action="store_true")
    parser.add_argument("--eval_per_step", type=int, default=None)
    parser.add_argument("--nmdn_eval", action="store_true", help="Use NMDN score as evaluation metric")
    #----------------------------------------------------------------------------------------------#

    #----------------------------------------Data Arguments----------------------------------------#
    # Training/Validation
    parser.add_argument('--data_provider', type=str)
    parser.add_argument("--dataset_name", type=str, default=None, help="The PYG file of the dataset")
    parser.add_argument("--dataset_names", type=str, action="append", help="additional PyG files to collate")
    parser.add_argument("--split", type=str, default=None, help="The split file of the dataset")
    parser.add_argument("--diffdock_nmdn_result", type=str, default=None, action="append", 
                        help="Use NMDN score to select input geometries for training.")
    parser.add_argument("--diffdock_confidence", action="store_true", help="Use diffdock-confidence score to select input geometries. overwrites --diffdock_nmdn_result")
    parser.add_argument("--valid_size", type=int, default=None, help="Validation size")
    parser.add_argument("--split_seed", type=int, default=2333, help="Seed for random splitting.")
    parser.add_argument('--data_root', type=str, default="../dataProviders/data")
    parser.add_argument('--remove_atom_ids', type=int, default=[], action="append", help='remove atoms from dataset')
    parser.add_argument("--add_sqf", type=str, action="append", default=[])
    parser.add_argument("--no_cut_protein", action="store_true")
    parser.add_argument("--cutoffs")
    parser.add_argument("--pl_cutoff", default=None, type=float)
    parser.add_argument("--proc_in_gpu", action="store_true",
        help="preprocess (distance matrix and edge calculation) in gpu. multiple workers does not work in this way so num_workers will be set to 0")
    parser.add_argument("--prot_embedding_root", default=None)
    parser.add_argument("--prot_embedding_roots", action="append", help="additional protein embeddings")
    parser.add_argument("--prot_embed_use_chunks", action="store_true",
        help="Prot embedding uses chunks behaviour: instead of one protein one file, use thousands of protein per file")
    parser.add_argument("--prot_info_ds", default=None, help="Use a separate dataset to store protein information (coordinates, PP bonds, etc...)")
    parser.add_argument("--atom_prop_ds", default=None, help="Use a separate dataset to store atom property for aux task or delta learning.")
    parser.add_argument("--cache_bonds", action="store_true", help="Cache bonds to avoid recomputatiob. It will comsume more memory.")
    parser.add_argument("--rmsd_csv", default=None, type=str, help="RMSD information")
    parser.add_argument("--rmsd_expansion", default=None, type=str, help="Expand RMSD similar to RBF.")
    parser.add_argument("--debug_mode_n_train", default=1000, type=int)
    parser.add_argument("--debug_mode_n_val", default=100, type=int)
    # Test
    parser.add_argument("--test_name", type=str, default=None, help="Specify the external test set.")
    parser.add_argument("--test_set", default=None)
    parser.add_argument("--proc_lit_pcba", action="store_true")
    # Dataloader
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--dynamic_batch", action="store_true")
    parser.add_argument("--dynamic_batch_max_num", type=int, default=None)
    parser.add_argument('--valid_batch_size', type=int)
    parser.add_argument("--over_sample", action="store_true")
    # KANO DS
    parser.add_argument("--kano_ds", default=None, type=str, help="pickle file saving all kano processed data set.")
    # Precomputed Atom/Mol Prop
    parser.add_argument("--lig_identifier_src", default="ligand_file", type=str, help="Names to choose as unique identifier")
    parser.add_argument("--lig_identifier_dst", default="ligand_file", type=str, help="Names to choose as unique identifier")
    parser.add_argument("--precomputed_mol_prop", action="store_true")
    # Precomputed LinF9 score
    parser.add_argument("--linf9_csv", default=None, type=str, help="RMSD information")
    #----------------------------------------------------------------------------------------------#

    #----------------------------------------Misc Arguments----------------------------------------#
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--debug_mode', type=str, default="False")
    parser.add_argument("--time_debug", type=str, default="False")
    parser.add_argument("--mem_debug", type=str, default="False")
    parser.add_argument("--chk", type=str, default=None)
    parser.add_argument('--log_file_name', type=str, default="training.log")
    parser.add_argument('--folder_prefix', type=str)
    parser.add_argument('--config_name', type=str, default='config.txt')
    parser.add_argument('--edge_version', type=str, help="voronoi | cutoff")
    parser.add_argument('--cutoff', type=float, default=10.)
    parser.add_argument('--boundary_factor', type=float, default=100.)
    parser.add_argument('--frag9_train_size', type=int, help="solely used for training curve")
    parser.add_argument('--frag20_train_size', type=int, help="solely used for training curve")
    parser.add_argument("--legacy_exps", action="store_true")
    parser.add_argument("--mdn2pkd_model", type=str, default=None, choices=["xgb", "rf", "linear"])
    parser.add_argument("--mdn_embed_type", default=None, type=str, choices=["nll", "prob"])
    parser.add_argument("--rmsd_threshold", type=float, default=None, 
        help="Predict if the structure is within a RMSD cutoff. It is used to train a DiffDock-like confidence model.")
    parser.add_argument("--mem", type=int, default=55, help="Only be parsed by smart_job_submit.py")
    #----------------------------------------------------------------------------------------------#
    return parser


# updated evidential regression loss
def evidential_loss_new(mu, v, alpha, beta, targets, lam=1, epsilon=1e-4):
    """
    Adapted from https://pubs.acs.org/doi/10.1021/acscentsci.1c00546
    Use Deep Evidential Regression negative log likelihood loss + evidential
        regularizer

    :mu: pred mean parameter for NIG
    :v: pred lam parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """
    # Calculate NLL loss
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)

    L_NLL = nll

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - mu))
    reg = error * (2 * v + alpha)
    L_REG = reg

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = L_NLL + lam * (L_REG - epsilon)

    return loss


def remove_handler(log=None):
    if log is None:
        log = logging.getLogger()
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    return


def fix_model_keys(state_dict):
    tmp = OrderedDict()
    for key in state_dict:
        if key.startswith("module."):
            # for some reason some module was saved with "module.module_list.*"
            tmp[key.split("module.")[-1]] = state_dict[key]
        elif key.startswith("module"):
            num = key.split(".")[0].split("module")[-1]
            tmp["main_module_list.{}.{}".format(num, ".".join(key.split(".")[1:]))] = state_dict[key]
        else:
            tmp[key] = state_dict[key]
    return tmp


def process_state_dict(state_dict: OrderedDict, config_dict: dict, logger, is_main=True):
    # this happens when loading checkpoints for SphereNet.
    # check https://github.com/divelab/DIG_storage/tree/main/3dgraph/qm9
    if "model_state_dict" in state_dict:
        og_state_dict = state_dict["model_state_dict"]
        new_state_dict = OrderedDict()
        for key in og_state_dict:
            new_state_dict["main_module_list.0." + key] = og_state_dict[key]
        state_dict = new_state_dict

    if config_dict["chk"]:
        if config_dict["reset_output_layers"]:
            logger.warn("WARNING: You are training from a checkpoint, you cannot reset output layers.")
        if config_dict["reset_scale_shift"]:
            logger.warn("WARNING: You are training from a checkpoint, you cannot reset scale or shift.")
        return state_dict

    if config_dict["reset_output_layers"] or config_dict["reset_scale_shift"]:
        # OrderedDict is immutable so I have to make a copy
        new_state_dict = OrderedDict()
        shift_reg = re.compile(r"shift.*")
        scale_reg = re.compile(r"scale.*")

        reset_list = []
        for ptn in config_dict["reset_ptn"]:
            reset_list.append(re.compile(ptn))
        if config_dict["reset_output_layers"]:
            logger.info("reset output layers...")
            # output layer for PhysNet
            reset_list.append(re.compile(r"main_module_list.*\.output\.lin\..*"))
            # output layer for ComENet
            reset_list.append(re.compile(r"main_module_list\..\.lin_out.*"))
            # output layer for SphereNet
            reset_list.append(re.compile(r"main_module_list\..\.init_v\.lin\.weight"))
            reset_list.append(re.compile(r"main_module_list\..\.update_vs\..\.lin\.weight"))
        if config_dict["reset_scale_shift"]:
            logger.info("reset scale and shift...")
            reset_list.append(shift_reg)
            reset_list.append(scale_reg)
        for key in state_dict:
            keep = True
            for reg in reset_list:
                if reg.fullmatch(key) is not None:
                    keep = False
                    if is_main:
                        logger.info(f"discarding: {key}")
                    break
            if keep:
                new_state_dict[key] = state_dict[key]
        return new_state_dict
    else:
        return state_dict


def validate_index(train_index: Union[List[int], torch.LongTensor], val_index, test_index):
    if isinstance(train_index, torch.LongTensor):
        train_index = train_index.tolist()
    if isinstance(val_index, torch.LongTensor):
        val_index = val_index.tolist()
    if isinstance(test_index, torch.LongTensor):
        test_index = test_index.tolist()
    # make sure the indexes are legit without overlapping, etc...
    train_size = len(train_index)
    train_index_set = set(train_index)
    assert train_size == len(train_index_set), f"{train_size}, {len(train_index_set)}"

    val_size = len(val_index)
    val_index_set = set(val_index)
    assert val_size == len(val_index_set), f"{val_size}, {len(val_index_set)}"
    assert len(train_index_set.intersection(val_index_set)) == 0, "You have a problem :)"

    if test_index is not None:
        test_size = len(test_index)
        test_index_set = set(test_index)
        assert test_size == len(test_index_set), f"{test_size}, {len(test_index_set)}"
        assert len(train_index_set.intersection(test_index_set)) == 0, "You have a problem :)"
    else:
        test_size = None

    return train_size, val_size, test_size


def mp_mean_std_calculate(ds, train_index, config, run_directory):
    # calculate the mean and standard deviation through multi-processing
    prop_tensor = []
    n_tensor = []
    __, __, num_workers = solv_num_workers()
    mp_fn = partial(_sp_mean_std, ds=ds, prop_names=config["target_names"])
    # mp_res = process_map(mp_fn, train_index.numpy().tolist(), chunksize=10, max_workers=num_workers+4)
    mp_res = [mp_fn(i) for i in tqdm(train_index.numpy().tolist())]
    for p, n in mp_res:
        prop_tensor.append(p)
        n_tensor.append(n)
    prop_tensor = torch.as_tensor(prop_tensor, dtype=floating_type)
    n_tensor = torch.as_tensor(n_tensor).long()
    torch.save(prop_tensor, osp.join(run_directory, "prop_tensor.pth"))
    torch.save(n_tensor, osp.join(run_directory, "n_tensor.pth"))
    mean_atom, std_atom = atom_mean_std(prop_tensor, n_tensor, torch.arange(len(train_index)))
    print(mean_atom)
    print(std_atom)
    return mean_atom, std_atom


def _sp_mean_std(i, ds, prop_names):
    this_data = ds.get(i, process=False)
    return ([getattr(this_data, p) for p in prop_names], this_data.N.item())


# https://github.com/faif/python-patterns/blob/master/patterns/creational/lazy_evaluation.py
def lazy_property(fn):
    """
    A lazy property decorator.

    The function decorated is called the first time to retrieve the result and
    then that calculated result is used the next time you access the value.
    """
    attr = "_lazy__" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr):
            setattr(self, attr, fn(self))
        return getattr(self, attr)

    return _lazy_property


class DistCoeCalculator:
    def __init__(self, pair_prop_dist_coe: Optional[str]) -> None:
        # calculate a coefficient based on the distance between AA and ligand atom
        if pair_prop_dist_coe is None: pair_prop_dist_coe = "identity"
        # "pair_prop_dist_coe" should be a series of operations connected by "_"
        # for example: inverse_square_norm will inverse distance, calculate square of it and perform normalization
        # You can compute multiple coefficients. For example, 'identity,inverse,inverse_square' 
        # will return a [-1, 3] dimension coefficient.
        self.op_lists: List[List[str]] = [coe_str.split("_") for coe_str in pair_prop_dist_coe.split(",")]
        self.coe_dim: int = len(self.op_lists)

    def __call__(self, pair_dist: torch.Tensor) -> torch.Tensor:
        coe_list: List[torch.Tensor] = [self.single_coe(pair_dist, ops) for ops in self.op_lists]
        coe = torch.concat(coe_list, dim=-1)
        return coe

    def single_coe(self, pair_dist: torch.Tensor, ops: List[str]) -> torch.Tensor:
        # compute coefficient based on a list of operations
        coe = pair_dist
        for op in ops:
            if op == "identity":
                coe = torch.ones_like(pair_dist.view(-1, 1))
            elif op == "inverse":
                coe = 1. / coe
            elif op == "square":
                coe = coe ** 2
            else:
                assert op == "norm", op
                coe = coe / coe.sum()
        assert isinstance(coe, torch.Tensor), coe.__class__
        return coe.view(-1, 1)


def torchdrug_imports():
    return
    from ocpmodels.models.equiformer_v2.edge_rot_mat import InitEdgeRotError
    # They have to be imported before torchdrug for some reason, otherwise they will fail
    import torchvision

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

if __name__ == '__main__':
    dummy_input = torch.rand(32, 160).double().cuda()
    dummy_model = torch.nn.Linear(160, 4).double().cuda()
    dummy_input = dummy_model(dummy_input)
    means, log_lambdas, log_alphas, log_betas = torch.split(dummy_input, dummy_input.shape[-1] // 4, dim=-1)
    soft_plus = torch.nn.Softplus()
    min_val = 1e-6
    lambdas = soft_plus(log_lambdas) + min_val
    # add 1 for numerical contraints of Gamma function
    alphas = soft_plus(log_alphas) + min_val + 1
    betas = soft_plus(log_betas) + min_val
    evi_cal_dict = {"mu": means, "v": lambdas, "alpha": alphas, "beta": betas}
    prop_pred = means
    evi_cal_dict["targets"] = torch.rand(32, 1).double().cuda()
    loss = evidential_loss_new(**evi_cal_dict).sum()
    loss.backward()
    print("finished")
