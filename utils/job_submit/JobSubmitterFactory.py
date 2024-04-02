import os.path as osp

from utils.job_submit.test_lit_pcba_submitters import LIT_PCBA_JobSubmitter
from utils.job_submit.test_prot_pep_submitters import ProtPepTestSubmitter
from utils.job_submit.regular_job_submitters import TemplateJobSubmitter, TestJobSubmitter, TrainJobSubmitter
from utils.job_submit.test_casf_job_submitters import CASF_BlindDockJobSubmitter, CASF_JobSubmitter


class JobSubmitterFactory:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def get_submitter(self, tgt: str) -> TemplateJobSubmitter:
        cfg = self.cfg
        debug = cfg["debug"]
        wait = cfg["wait"]
        if not osp.isdir(tgt):
            # Training job from config file
            assert osp.basename(tgt).startswith("config-")
            return TrainJobSubmitter(tgt, debug)

        # Testing job from a trained directory
        if cfg["lit_pcba_diffdock"] or cfg["lit_pcba_diffdock_nmdn"]:
            return LIT_PCBA_JobSubmitter(tgt, debug, cfg["ref"], wait,
                        cfg["target"], cfg["lit_pcba_diffdock_nmdn"])
        elif cfg["casf_diffdock"]:
            return CASF_BlindDockJobSubmitter(tgt, debug, cfg["ref"], wait)
        elif osp.basename(tgt.rstrip("/")).startswith(("exp_ppep_", "exp_pp_")):
            return ProtPepTestSubmitter(debug, tgt)
        return self.parse_submitter_from_args(tgt)

    def parse_submitter_from_args(self, tgt: str) -> TemplateJobSubmitter:
        cfg = self.cfg
        debug = cfg["debug"]
        wait = cfg["wait"]

        ds_args: dict = TestJobSubmitter(tgt, True, False, True).ds_args
        ds_name = osp.basename(ds_args["file_locator"] if "file_locator" in ds_args else ds_args["dataset_name"])
        is_testing_casf = False
        for head in ["PL_train-", "PDBind_v2020", "PBind2020", "biolip."]:
            if ds_name.startswith(head): is_testing_casf = True
        if is_testing_casf:
            return CASF_JobSubmitter(tgt, debug, cfg["ref"], wait)
        return TestJobSubmitter(tgt, debug, cfg["ref"], wait)