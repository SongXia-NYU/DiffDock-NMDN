import argparse

from utils.scores.merck_fep_scores import MerckFEPScoreCalculator


parser = argparse.ArgumentParser()
parser.add_argument("--folder_names")
parser.add_argument("--diffdock", action="store_true")
args = parser.parse_args()
calculator = MerckFEPScoreCalculator(args.folder_names, {}, args.diffdock)
calculator.run()
