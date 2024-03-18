from collections import Counter, defaultdict
import json
import os
import os.path as osp
import subprocess
from typing import List
from tqdm import tqdm
import pandas as pd
from glob import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

class CASP14Reader:
    # 427 is the group number of AlphaFold2
    def __init__(self, ds_root, group_num=427) -> None:
        assert group_num == 427, "Currently only support AlphaFold2 results!"
        self.ds_root = ds_root
        self.eval_by_target_root = osp.join(ds_root, "eval_results", "by_target")
        self.eval_by_model_root = osp.join(ds_root, "eval_results", "by_model")
        self.preds_root = osp.join(ds_root, "predictions", "regular")
        self.preds_domain_root = osp.join(ds_root, "predictions_trimmed_to_domains")
        self.group_num = group_num
        self.gnum_fmt = str(self.group_num).zfill(3)

        self._tgt_list = None
        self._eval_df = None
        self.eval_csv = osp.join(self.eval_by_model_root, f"{group_num}.casp14_eval.csv")
        self._recalc_score_df = None
        self.recalc_score_csv = osp.join(self.eval_by_model_root, f"{group_num}.recalc_tm.csv")
        print(f"Recalc score csv: {self.recalc_score_csv}")
        self._esm2_targets = None
        self._esm2_domain_targets = None

    def get_prediction(self, tgt_name, rank=1, domain=None):
        droot = self.preds_root if domain is None else self.preds_domain_root
        ext = ""
        if domain is not None:
            ext += f"-D{domain}"
        return osp.join(droot, tgt_name+ext, f"{tgt_name}TS{self.gnum_fmt}_{rank}{ext}.pdb")
    
    def get_target(self, tgt_name, domain=None):
        fname = f"{tgt_name}.pdb" if domain is None else f"{tgt_name}-D{domain}.pdb"
        if tgt_name in ["T1045s2", "T1055", "T1058", "T1060s2", "T1060s3", "T1061", "T1070", 
                        "T1076", "T1078", "T1089", "T1091"]:
            return osp.join(self.ds_root, "targets", "amend", fname)
        
        return osp.join(self.ds_root, "targets", fname)
    
    @property
    def recalc_score_df(self):
        if self._recalc_score_df is not None:
            return self._recalc_score_df
        elif True and osp.exists(self.recalc_score_csv):
            self._recalc_score_df = pd.read_csv(self.recalc_score_csv)
            return self._recalc_score_df
        
        info_dict = defaultdict(lambda: [])
        failed_targets = []
        for target_raw in tqdm(self.esm2_domain_targets):
            # T1024-D1 ---> target=T1024, domain=1
            splits = target_raw.split("-D")
            if len(splits) == 1:
                target = splits[0]
                domain = None
            else:
                target, domain = splits

            if target in ["T1026", "T1032"]:
                domain = None

            pred_pdb = self.get_prediction(target, 1, domain)
            target_pdb = self.get_target(target, domain)
            if not osp.exists(target_pdb) and domain=="1":
                target_pdb = self.get_target(target, None)

            tm_score_info = cal_tm_score(pred_pdb, target_pdb)
            if not tm_score_info:
                failed_targets.append(target_raw)
                continue

            lddt_score_info = lddt_score(pred_pdb, target_pdb)
            if not lddt_score_info:
                failed_targets.append(target_raw)
                continue

            info_dict["target"].append(target_raw)
            for key in tm_score_info:
                info_dict[key].append(tm_score_info[key])
            for key in lddt_score_info:
                info_dict[key].append(lddt_score_info[key])
        info_df = pd.DataFrame(info_dict)
        info_df.to_csv(self.recalc_score_csv, index=False)
        self._recalc_score_df = info_df
        print(f"Failed targets: {failed_targets}")
        return self._recalc_score_df
    
    @property
    def tgt_list(self):
        if self._tgt_list is not None:
            return self._tgt_list
        
        tgt_list = []
        for f in glob(osp.join(self.ds_root, "targets", "T*.pdb")):
            tgt_list.append(osp.basename(f).split(".pdb")[0])
        self._tgt_list = tgt_list
        return self._tgt_list
    
    @property
    def eval_df(self):
        if self._eval_df is not None:
            return self._eval_df
        elif osp.exists(self.eval_csv):
            eval_df = pd.read_csv(self.eval_csv)
            # eval_df = clear_d1(eval_df)
            self._eval_df = eval_df
            return self._eval_df
        
        by_target_csvs = glob(osp.join(self.eval_by_target_root, "*.csv"))
        dfs = []
        for csv in by_target_csvs:
            target = osp.basename(csv).split(".csv")[0]
            info_df = pd.read_csv(csv, header=1)
            info_df = info_df[info_df["GR#"] == str(self.group_num)]
            if info_df.shape[0] == 0:
                continue
            
            info_df["target"] = target
            dfs.append(info_df)
        eval_df = pd.concat(dfs)
        eval_df.to_csv(self.eval_csv, index=False)
        self._eval_df = eval_df
        return self._eval_df
        
    
    def gather_eval_results(self):
        """
        Download evaluation results from the CASP14 website.
        """
        targets = self.get_target_list()
        url_tmpl = "https://predictioncenter.org/casp14/results.cgi?view=tables&target={target}&model=1&groups_id="

        for target in targets:
            res = pd.read_html(url_tmpl.format(target=target))
            for df in res:
                if df.shape[0] <= 20:
                    continue
                df.columns = df.iloc[0]
                df = df[1:]
                df.to_csv(osp.join(self.eval_by_target_root, f"{target}.csv"), index=False)
    
    def get_target_list(self) -> List[str]:
        """
        Get the list of total number of targets from the CASP14 website.
        """
        def _filter_info(res_df: pd.DataFrame):
            try:
                info_str = res_df.iloc[0, 0]
            except IndexError:
                return None
            
            if not isinstance(info_str, str):
                return None
            
            if not info_str.startswith("T"):
                return None
            
            info_str = info_str.split()[0]
            patterns = [re.compile(r"T...."), re.compile(r"T....-D."), re.compile(r"T....s.-D."), re.compile(r"T....s.")]
            for ptn in patterns:
                if ptn.fullmatch(info_str):
                    return info_str
            return None
        
        def _find_targets_url(url: str):
            res = pd.read_html(url)
            out = []
            for r in res:
                tgt = _filter_info(r)
                if tgt:
                    out.append(tgt)
            return out

        urls = ["https://predictioncenter.org/casp14/results.cgi",
                "https://predictioncenter.org/casp14/results.cgi?tr_type=all&offset=T1054&",
                "https://predictioncenter.org/casp14/results.cgi?tr_type=all&offset=T1084&"]
        targets = []
        for url in urls:
            targets.extend(_find_targets_url(url))
        return targets
    
    def exp_grp_preds_domain(self):
        """
        Extract the compressed files for a specific group's predictions by domain and rename them
        """
        wildcards = f"T*/T*TS{self.gnum_fmt}*"
        for f in glob(osp.join(self.preds_domain_root, "*.tar.gz")):
            cmd = f"cd {self.preds_domain_root}; tar xvf {osp.basename(f)} --wildcards '{wildcards}'"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e)

        for f in glob(osp.join(self.preds_domain_root, "T????-D?", "T????TS???_?-D?")):
            os.rename(f, f+".pdb")
        for f in glob(osp.join(self.preds_domain_root, "T????s?-D?", "T????s?TS???_?-D?")):
            os.rename(f, f+".pdb")

    def temp(self):
        for f in glob(osp.join(self.preds_domain_root, "T????-D?", "T????TS???_?-D?")):
            os.rename(f, f+".pdb")

    def extract_grp_preds(self):
        """
        Extract the compressed files for a specific group's predictions and rename them
        """
        wildcards = f"T*/T*TS{self.gnum_fmt}*"
        for f in glob(osp.join(self.preds_root, "*.tar.gz")):
            cmd = f"cd {self.preds_root}; tar xvf {osp.basename(f)} --wildcards '{wildcards}'"
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(e)

        for f in glob(osp.join(self.preds_root, "T????", "T????TS???_?")):
            os.rename(f, f+".pdb")
        for f in glob(osp.join(self.preds_root, "T????s?", "T????s?TS???_?")):
            os.rename(f, f+".pdb")

    def download_targets(self):
        """
        use `wget -r --no-parent https://predictioncenter.org/casp14/TARGETS_PDB/`
        """
        raise NotImplementedError
    
    @property
    def esm2_targets(self):
        if self._esm2_targets is not None:
            return self._esm2_targets

        esm2_target_json = osp.join(osp.dirname(__file__), "ESM2-CASP14-targets.json")
        with open(esm2_target_json) as f:
            self._esm2_targets = json.load(f)
        return self._esm2_targets
    
    @property
    def esm2_domain_targets(self):
        if self._esm2_domain_targets is not None:
            return self._esm2_domain_targets
        
        out = []
        eval_res_targets = self.eval_df["target"].values
        for target in self.esm2_targets:
            if target in eval_res_targets:
                out.append(target)
                continue

            out.append(target + "-D1")
        self._esm2_domain_targets = out
        return self._esm2_domain_targets


TM_SCORE = "/scratch/sx801/softwares/TM-score/TMscore"
def cal_tm_score(pred: str, target: str):
    cmd = f"{TM_SCORE} -seq {pred} {target}"
    score_out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return read_tm_score_out(score_out.stdout)

TM_ALIGN = "/scratch/sx801/softwares/TM-score/TMalign"
def cal_tm_score_align(pred: str, target: str):
    cmd = f"{TM_SCORE} -seq {pred} {target}"
    score_out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return read_tm_score_out(score_out.stdout)

def read_tm_score_out(score_out: str):
    out = {}
    for line in score_out.split("\n"):
        line = line.strip()
        if line.startswith("Structure1:"):
            out["len_structure1"] = int(line.split()[-1])
        elif line.startswith("Structure2:"):
            out["len_structure2"] = int(line.split("(")[0].split()[-1])
        elif line.startswith("Number of residues in common"):
            out["n_res_common"] = int(line.split()[-1])
        elif line.startswith("RMSD of  the common residues"):
            out["rmsd_res_common"] = float(line.split()[-1])
        elif line.startswith("TM-score"):
            out["tm_score"] = float(line.split("=")[1].split()[0])
        elif line.startswith("MaxSub-score"):
            out["maxsub_score"] = float(line.split("=")[1].split()[0])
    return out if out else None

def lddt_score(pred: str, target: str):
    # return fake lddt score to reduce computational cost
    # return {"global_lddt_score": 0.}
    cmd = f"lddt {pred} {target}"
    try:
        score_out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        print(e)
        return {"global_lddt_score": None}
    out = {}
    for line in score_out.split("\n"):
        line = line.strip()
        if line.startswith("Global LDDT score:"):
            out["global_lddt_score"] = float(line.split()[-1])
    return out if out else None

def test_run():
    reader = CASP14Reader("/vast/sx801/geometries/CASP14")
    reader.gather_eval_results()

def clear_d1(eval_df: pd.DataFrame):
    raise ValueError
    # a temporary solution assmuing a target with one domain is the target itself
    dom_counter = Counter()
    for i in range(eval_df.shape[0]):
        this_df = eval_df.iloc[i]
        this_target = this_df["target"]
        dom_counter.update([this_target.split("-D")[0]])
    aux_dfs = []
    for i in range(eval_df.shape[0]):
        this_df = eval_df.iloc[i]
        this_target = this_df["target"].split("-D")[0]
        if dom_counter[this_target] == 1:
            wanted_target = this_df["target"].split("-D")[0]
            this_df = eval_df.iloc[[i]]
            this_df["target"] = wanted_target
            aux_dfs.append(this_df)
    aux_dfs.append(eval_df)
    eval_df = pd.concat(aux_dfs)
    return eval_df
    
if __name__ == "__main__":
    test_run()
