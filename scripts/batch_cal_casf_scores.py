# Do NOT delete these two unused imports
# They have to be imported before torchdrug for some reason, otherwise they will fail
# import torchvision
# from ocpmodels.models.equiformer_v2.edge_rot_mat import InitEdgeRotError

import argparse
import os.path as osp
from glob import glob

from utils.scores.casf_blind_scores import CASFBlindDockScore, CASFBlindScreenScore
from utils.scores.casf_custom_screen import CASF_CustomScreen
from utils.scores.casf_scores import CasfScoreCalculator


def batch_cal_casf_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_names")
    parser.add_argument("--ref", action="store_true")
    parser.add_argument("--blind-dock", action="store_true")
    parser.add_argument("--blind-screen", action="store_true")
    parser.add_argument("--docking_config", default=None, type=str)
    parser.add_argument("--screening_config", default=None, type=str)
    parser.add_argument("--custom_screen", action="store_true")
    cfg = parser.parse_args()
    cfg: dict = vars(cfg)

    folders = glob(cfg["folder_names"])
    del cfg["folder_names"]

    blind_dock = cfg["blind_dock"]
    del cfg["blind_dock"]
    blind_screen = cfg["blind_screen"]
    del cfg["blind_screen"]
    custom_screen = cfg["custom_screen"]; del cfg["custom_screen"]
    
    for folder in folders:
        print(f"calculating CASF scores for {folder}")
        scorer_cls = CasfScoreCalculator
        if blind_dock: scorer_cls = CASFBlindDockScore
        if blind_screen: scorer_cls = CASFBlindScreenScore
        if custom_screen: scorer_cls = CASF_CustomScreen
        scorer = scorer_cls(osp.abspath(folder), cfg)
        scorer.run()


if __name__ == '__main__':
    batch_cal_casf_scores()
