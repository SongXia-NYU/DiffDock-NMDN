import argparse
import torchvision

from Networks.esm_gearnet.pl_dataset import ESMGearNetProtLig

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id: int = args.array_id

ds_args = {"data_root": "/scratch/sx801/data/im_datasets/",
           "dataset_name": "PBind2020OG.hetero.polar.polar.implicit.min_dist.pyg",
           "split": "PBind2020OG.hetero.polar.polar.implicit.min_dist.covbinder.res2.5.testsets.poslinf9.metal.pth",
           "test_name": "hetero.polar.polar.implicit.min_dist",
           "config_args": None}
ds = ESMGearNetProtLig(array_id, **ds_args)
breakpoint()
