import torch
import argparse
import os
import os.path as osp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_format", default="../../tmp/rand_splits/splits_lipop_rand_{}.pt")
    parser.add_argument("--num_splits", type=int, default=50)
    parser.add_argument("--train_size", type=int, default=3360)
    parser.add_argument("--valid_size", type=int, default=420)
    parser.add_argument("--test_size", type=int, default=420)

    args = parser.parse_args()
    for i in range(args.num_splits):
        splits = ["train", "valid", "test"]
        total_size = 0
        for split in splits:
            total_size += getattr(args, f"{split}_size")

        perm = torch.randperm(total_size)
        split_dict = dict()
        size_prev = 0
        for split in splits:
            split_dict[f"{split}_index"] = perm[size_prev: size_prev+getattr(args, f"{split}_size")]
            size_prev += getattr(args, f"{split}_size")

        save_path = args.save_format.format(i)
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        torch.save(split_dict, save_path)
