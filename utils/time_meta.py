import time
from treelib import Tree
import torch
import os.path as osp

'''
This global dictionary will record run time info while running, which may help find the bottle neck
'''
function_and_time = {}
function_and_count = {}


def record_data(name, t0, syn=False):
    if torch.cuda.is_available() and syn:
        torch.cuda.synchronize()
    delta_t = time.time() - t0
    if name in function_and_time.keys():
        function_and_time[name] += delta_t
        function_and_count[name] += 1
    else:
        function_and_time[name] = delta_t
        function_and_count[name] = 1
    return time.time()


def print_function_runtime(folder="."):
    for f_name, info_dict in zip(["meta_time.txt", "meta_count.txt"], [function_and_time, function_and_count]):
        tree = Tree()
        tree.create_node("ROOT", "root")
        train_root_id = "root"
        if "individual_runs" in info_dict.keys():
            tree.create_node("individual_runs_{}".format(info_dict["individual_runs"]), "individual_runs",
                             parent="root")
            train_root_id = "individual_runs"
        for name in ["setup", "training", "collate_fn"]:
            tree.create_node("{}_{}".format(name, info_dict[name]), name, parent=train_root_id)
        for name in ["forward", "loss_cal", "backward", "step"]:
            tree.create_node("{}_{}".format(name, info_dict[name]), name, parent="training")
        for name in ["bond_setup", "msg_bond_setup", "expansion_prepare", "embedding_prepare", "main_modules",
                     "normalization", "post_modules", "scatter_pool_others"]:
            tree.create_node("{}_{}".format(name, info_dict[name]), name, parent="forward")
        for name in ["validate_simple", "select_index", "save_chk", "al_init_setup"]:
            if name in info_dict.keys():
                tree.create_node("{}_{}".format(name, info_dict[name]), name, parent="root")
        tree.save2file(osp.join(folder, f_name))

    return function_and_time
