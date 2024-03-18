import argparse
import os.path as osp
from glob import glob

from utils.scores.metallo_protein_scores import MetalloProteinScores



def batch_metal_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_names")
    args = parser.parse_args()
    folders = glob(args.folder_names)
    args = vars(args)
    del args["folder_names"]
    for folder in folders:
        print(f"calculating metalloprotein scores for {folder}")
        scorer_cls = MetalloProteinScores

        scorer = scorer_cls(osp.abspath(folder))
        scorer.run()


if __name__ == '__main__':
    batch_metal_scores()
