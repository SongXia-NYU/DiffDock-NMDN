# Do NOT delete these two unused imports
# They have to be imported before torchdrug for some reason, otherwise they will fail
import torchvision
from ocpmodels.models.equiformer_v2.edge_rot_mat import InitEdgeRotError

from argparse import ArgumentParser
from typing import List

from utils.job_submit.regular_job_submitters import JobSubmitter
from utils.job_submit.JobSubmitterFactory import JobSubmitterFactory

class BatchJobSubmitter:
    def __init__(self, cfg: dict) -> None:
        targets = cfg["targets"]
        factory = JobSubmitterFactory(cfg)
        self.job_submitters: List[JobSubmitter] = [factory.get_submitter(i) for i in targets]

    def run(self):
        for submitter in self.job_submitters: submitter.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ref", action="store_true")
    parser.add_argument("--casf-diffdock", action="store_true")
    parser.add_argument("--lit-pcba", action="store_true")
    parser.add_argument("--lit-pcba-diffdock", action="store_true")
    parser.add_argument("--target", default=None, help="None for all targets.")
    parser.add_argument("--wait", action="store_true", help="For testing jobs only. Submit the test job when the training job completes.")
    cfg = parser.parse_args()
    cfg: dict = vars(cfg)

    submitter = BatchJobSubmitter(cfg)
    submitter.run()


if __name__ == "__main__":
    main()
