from geometry_processors.pl_dataset.lit_pcba_reader import TARGETS

from argparse import ArgumentParser
from typing import List

from utils.job_submit.regular_job_submitters import JobSubmitter
from utils.job_submit.JobSubmitterFactory import JobSubmitterFactory

class BatchJobSubmitter:
    def __init__(self, cfg: dict) -> None:
        entities = cfg["entities"]
        factory = JobSubmitterFactory(cfg)

        if cfg["target"] == "all-separate-jobs":
            assert len(entities) == 1, entities
            self.job_submitters = []
            for target in TARGETS:
                factory.cfg["target"] = target
                self.job_submitters.append(factory.get_submitter(entities[0]))
        else:
            self.job_submitters: List[JobSubmitter] = [factory.get_submitter(i) for i in entities]

    def run(self):
        for submitter in self.job_submitters: submitter.run()


def main():
    parser = ArgumentParser()
    parser.add_argument("entities", nargs="+")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ref", action="store_true")
    parser.add_argument("--casf-diffdock", action="store_true")
    parser.add_argument("--lit-pcba-diffdock", action="store_true")
    parser.add_argument("--lit-pcba-diffdock-nmdn", action="store_true")
    parser.add_argument("--target", default=None, help="default is for all targets.")
    parser.add_argument("--wait", action="store_true", help="For testing jobs only. Submit the test job when the training job completes.")
    cfg = parser.parse_args()
    cfg: dict = vars(cfg)

    submitter = BatchJobSubmitter(cfg)
    submitter.run()


if __name__ == "__main__":
    main()
