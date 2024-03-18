import argparse
import os.path as osp
from glob import glob
from utils.scores.lit_pcba_custom_screen import LIT_PCBA_CustomWrapper

from utils.scores.lit_pcba_screening import LIT_PCBA_ScreeningWrapper
from utils.scores.lit_pcba_summarizer import LIT_PCBA_Summarizer


def batch_cal_lit_pcba():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_names")
    parser.add_argument("--target")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    folders = glob(args.folder_names)
    args = vars(args)
    del args["folder_names"]
    for folder in folders:
        print(f"calculating Screening scores for {folder}")
        scorer_cls = LIT_PCBA_ScreeningWrapper
        if args["custom"]: scorer_cls = LIT_PCBA_CustomWrapper

        if args["summarize"]:
            LIT_PCBA_Summarizer(osp.abspath(folder)).run()
            continue

        scorer = scorer_cls(osp.abspath(folder), args["target"])
        scorer.run()


if __name__ == '__main__':
    batch_cal_lit_pcba()
