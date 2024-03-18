import argparse
from utils.scores.prot_pep_scores import ProtPepScoreCalculator


def batch_cal_ppep_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", default="exp_ppep_001_run_2023-05-02_161649__691908")
    args = parser.parse_args()
    calculator = ProtPepScoreCalculator(args.folder_name)
    calculator.run()


if __name__ == "__main__":
    batch_cal_ppep_scores()
