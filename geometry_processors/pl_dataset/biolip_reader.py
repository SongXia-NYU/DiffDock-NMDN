import copy
from glob import glob
import json
import math
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import yaml
from tqdm.contrib.concurrent import process_map

from geometry_processors.filter.covalent_binder import CovalentBondFilter
from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
from geometry_processors.pl_dataset.lit_pcba_reader import LIT_PCBA_Reader
from geometry_processors.pl_dataset.prot_utils import pdb2seq

class BioLipReader:
    def __init__(self, droot: str) -> None:
        # folders
        self.droot = droot
        self.lig_root = osp.join(droot, "ligand")
        self.lig_polar_root = osp.join(droot, "ligand_polar")
        self.lig_pdbqt_root = osp.join(droot, "ligand_pdbqt")
        self.lig_linf9_root = osp.join(droot, "ligand_linf9")
        self.rec_root = osp.join(droot, "receptor")
        self.prot_polar_root = osp.join(droot, "receptor_polar")
        self.info_root = osp.join(droot, "info")
        self.vis_root = osp.join(droot, "visualization")
        self.prot_cif_root = osp.join(droot, "protein_cif")

        # files
        self.pl_info_csv = osp.join(self.info_root, "biolip_prot_lig.csv")

        self._pl_info_df = None
        # Level one filter: passing LinF9 optimization
        self._pl_lvl_1_info_df = None
        # Level two filter: Level one + non-covalent binders only
        self._pl_lvl_2_info_df = None
        # Level three filter: Level two + remove negative LinF9 scores
        self._pl_lvl_3_info_df = None
        # Level four filter: Level three + remove low resolution structures
        self._pl_lvl_4_info_df = None
        # Level final filter: Removing examples from CASF/DiffDock
        self._pl_lvl_final_info_df = None
        # Extract pKd information from verbose
        self._pkd_info_df = None
        self._pbind_info_df = None
        self._stats = None
        self._unique_pdb_ids = None
        self._pdb2cbr = None
        self._polar_prot2seq_info = None

    @property
    def polar_prot2seq_info(self) -> Dict[str, str]:
        if self._polar_prot2seq_info is not None:
            return self._polar_prot2seq_info
        
        cache_yaml = osp.join(self.info_root, "polar_prot2seq.yaml")
        if osp.exists(cache_yaml):
            with open(cache_yaml) as f:
                self._polar_prot2seq_info = yaml.safe_load(f)
            return self._polar_prot2seq_info
        
        polar_prot2seq_info = {}
        job_list = []
        fl_list = []
        added_fl = set()
        for i in tqdm.tqdm(range(self.pl_lvl_final_info_df.shape[0])):
            this_info = self.pl_lvl_final_info_df.iloc[i]
            prot, __ = self.get_polar_pl(None, this_info)
            fl = osp.basename(prot).split(".pdb")[0]
            if fl in added_fl or not osp.exists(prot):
                continue

            added_fl.add(fl)
            fl_list.append(fl)
            job_list.append(prot)
            
        seq_list = process_map(pdb2seq, job_list, max_workers=14, chunksize=10)
        for fl, seq in zip(fl_list, seq_list):
            polar_prot2seq_info[fl] = seq

        with open(cache_yaml, "w") as f:
            yaml.safe_dump(polar_prot2seq_info, f)
        self._polar_prot2seq_info = polar_prot2seq_info
        return self._polar_prot2seq_info

    @property
    # Level final filter: Removing examples from CASF/DiffDock
    def pl_lvl_final_info_df(self):
        if self._pl_lvl_final_info_df is not None:
            return self._pl_lvl_final_info_df
        
        # Cache on disk
        info_csv = osp.join(self.info_root, "pl_lvl_final_info.csv")
        if osp.exists(info_csv):
            self._pl_lvl_final_info_df = pd.read_csv(info_csv)
            return self._pl_lvl_final_info_df
        
        # PDB ids that are in the test set.
        # Remove CASF PDBs
        casf_reader = CASF2016Reader("/vast/sx801/geometries/CASF-2016-cyang")
        casf_pdbs = set(casf_reader.pdbs)
        # Remove DiffDock PDBs
        diffdock_csv = "/scratch/sx801/scripts/DiffDock/data/testset_csv.csv"
        diffdock_pdbs = set(pd.read_csv(diffdock_csv)["complex_name"].values.tolist())
        masked_pdbs = casf_pdbs.union(diffdock_pdbs)
        # Remove LIT-PCBA PDBs
        PCBA_ROOT = "/vast/sx801/geometries/LIT-PCBA"
        for f in glob(osp.join(PCBA_ROOT, "*")):
            tgt_name = osp.basename(f)
            pcba_reader = LIT_PCBA_Reader(PCBA_ROOT, tgt_name)
            masked_pdbs = masked_pdbs.union(set(pcba_reader.prot_pdbs()))

        prev_df = self.pl_lvl_4_info_df
        masker = []
        for i in range(prev_df.shape[0]):
            this_info = prev_df.iloc[i]
            this_pdb = this_info["pdb_id"]
            if this_pdb in masked_pdbs:
                masker.append(False)
            else:
                masker.append(True)
        res_df = prev_df[masker]
        res_df = self.add_uuid_col(res_df)
        res_df.to_csv(info_csv, index=False)

        self._pl_lvl_final_info_df = res_df
        return self._pl_lvl_final_info_df
    
    @property
    def pdb2cbr(self):
        if self._pdb2cbr is not None:
            return self._pdb2cbr
        
        cache_cbr = osp.join(self.info_root, "pdb2cbr.nucleophile.binder.json")
        if osp.exists(cache_cbr):
            with open(cache_cbr) as f:
                self._pdb2cbr = json.load(f)
            return self._pdb2cbr
        
        res = {}
        filterer = CovalentBondFilter(None, False, set(["acceptable_nucleophile", "invalid_binder"]))
        for target in tqdm.tqdm(self.unique_pdb_ids):
            cbr = filterer.get_covalent_bond_record(target)
            res[target] = cbr
        with open(cache_cbr, "w") as f:
            json.dump(res, f, indent=2)
        self._pdb2cbr = res
        return self._pdb2cbr

    @property
    def unique_pdb_ids(self) -> List[str]:
        if self._unique_pdb_ids is not None:
            return self._unique_pdb_ids
        
        cache_json = osp.join(self.info_root, "unique_pdb_ids.json")
        if osp.exists(cache_json):
            with open(cache_json) as f:
                self._unique_pdb_ids = json.load(f)
            return self._unique_pdb_ids
        
        unique_pdb_ids = set(self.pl_info_df["pdb_id"].values.tolist())
        for pdb_id in unique_pdb_ids:
            assert len(pdb_id) == 4, pdb_id
        unique_pdb_ids = list(unique_pdb_ids)
        with open(cache_json, "w") as f:
            json.dump(unique_pdb_ids, f, indent=2)
        self._unique_pdb_ids = unique_pdb_ids
        return self._unique_pdb_ids

    @property
    def pl_info_df(self) -> pd.DataFrame:
        if self._pl_info_df is not None:
            return self._pl_info_df
        
        # columns information available at https://zhanggroup.org/BioLiP/download/readme.txt
        cols = ["pdb_id", "receptor_chain", "resolution", "binding_site_num", "lig_id_ccd", "lig_chain",
                "lig_serial_num", "binding_site_res", "binding_site_res_renum", "catalytic_site_res", 
                "catalytic_site_res_renum", "ec_num", "go_terms", "binding_affinity_manual", 
                "binding_affinity_moad", "binding_affinity_pdbbind", "binding_affinity_bindingdb",
                "uniprot_id", "pubmed_id", "lig_res_seq_num", "receptor_seq"]
        info_df = pd.read_csv(self.pl_info_csv, header=0, sep="\t", names=cols)

        self._pl_info_df = info_df
        return self._pl_info_df
    
    @property
    # Level one filter: passing LinF9 optimization
    def pl_lvl_1_info_df(self):
        # Cache in memory
        if self._pl_lvl_1_info_df is not None:
            return self._pl_lvl_1_info_df
        
        # Cache on disk
        info_csv = osp.join(self.info_root, "pl_lvl_1_info.csv")
        if osp.exists(info_csv):
            self._pl_lvl_1_info_df = pd.read_csv(info_csv)
            return self._pl_lvl_1_info_df
        
        dfs = []
        for i in tqdm.tqdm(range(self.pl_info_df.shape[0])):
            this_info = self.pl_info_df.iloc[i]
            __, lig_linf9 = self.get_linf9_pl(None, this_info)
            if not osp.exists(lig_linf9):
                continue

            this_df = copy.deepcopy(self.pl_info_df.iloc[[i]])
            with open(lig_linf9) as f:
                for line in f.readlines():
                    if line.startswith("REMARK minimizedAffinity"):
                        this_df["minimizedAffinity"] = float(line.split()[-1])
                    elif line.startswith("REMARK minimizedRMSD"):
                        this_df["minimizedRMSD"] = float(line.split()[-1])
                        break
            dfs.append(this_df)
        res_df = pd.concat(dfs)
        
        self._pl_lvl_1_info_df = res_df
        self._pl_lvl_1_info_df.to_csv(info_csv, index=False)
        return self._pl_lvl_1_info_df

    @property
    # Level two filter: Level one + non-covalent binders only
    def pl_lvl_2_info_df(self):
        if self._pl_lvl_2_info_df is not None:
            return self._pl_lvl_2_info_df
        
        # Cache on disk
        info_csv = osp.join(self.info_root, "pl_lvl_2_info.csv")
        if osp.exists(info_csv):
            self._pl_lvl_2_info_df = pd.read_csv(info_csv)
            return self._pl_lvl_2_info_df
        
        non_cov_pdbs = set([pdb.lower() for pdb in self.pdb2cbr if not self.pdb2cbr[pdb]])
        dfs = []
        for i in tqdm.tqdm(range(self.pl_lvl_1_info_df.shape[0])):
            this_info = self.pl_lvl_1_info_df.iloc[i]
            pdb = this_info["pdb_id"]
            if pdb in non_cov_pdbs:
                dfs.append(self.pl_lvl_1_info_df.iloc[[i]])
        res_df = pd.concat(dfs)

        self._pl_lvl_2_info_df = res_df
        self._pl_lvl_2_info_df.to_csv(info_csv, index=False)
        return self._pl_lvl_2_info_df

    @property
    # Level three filter: Level two + remove negative LinF9 scores
    def pl_lvl_3_info_df(self):
        if self._pl_lvl_3_info_df is not None:
            return self._pl_lvl_3_info_df
        
        # Cache on disk
        info_csv = osp.join(self.info_root, "pl_lvl_3_info.csv")
        if osp.exists(info_csv):
            self._pl_lvl_3_info_df = pd.read_csv(info_csv)
            return self._pl_lvl_3_info_df
        
        fig, ax = plt.subplots(1, 1)
        sns.histplot(self.pl_lvl_2_info_df["minimizedAffinity"])
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_root, "linf9_dist.png"))
        plt.close()
        fig, ax = plt.subplots(1, 1)
        pbind_linf9 = self.to_pbind_df(self.pl_lvl_2_info_df)["minimizedAffinity"]
        sns.histplot(pbind_linf9)
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_root, "linf9_dist.pbind.png"))
        plt.close()
        
        prev_df = self.pl_lvl_2_info_df
        self._pl_lvl_3_info_df = prev_df[prev_df["minimizedAffinity"]<0]
        self._pl_lvl_3_info_df.to_csv(info_csv, index=False)
        return self._pl_lvl_3_info_df
    
    @property
    # Level four filter: Level three + remove low resolution structures
    def pl_lvl_4_info_df(self):
        if self._pl_lvl_4_info_df is not None:
            return self._pl_lvl_4_info_df
        
        # Cache on disk
        info_csv = osp.join(self.info_root, "pl_lvl_4_info.csv")
        if osp.exists(info_csv):
            self._pl_lvl_4_info_df = pd.read_csv(info_csv)
            return self._pl_lvl_4_info_df
        
        prev_df = self.pl_lvl_3_info_df
        res_df = prev_df[prev_df["resolution"] < 2.5]
        self._pl_lvl_4_info_df = res_df
        res_df.to_csv(info_csv, index=False)
        return self._pl_lvl_4_info_df


    @staticmethod
    def to_pbind_df(df: pd.DataFrame):
        return df[~ df["binding_affinity_pdbbind"].isna()]
    
    def uuid2pkd(self, uuid: str) -> Optional[float]:
        if uuid not in self.pkd_info_df.index:
            return None
        
        return self.pkd_info_df.loc[uuid, "mean_pkd"].item()
    
    @property
    def pkd_info_df(self) -> pd.DataFrame:
        if self._pkd_info_df is not None:
            return self._pkd_info_df
        
        pkd_csv = osp.join(self.info_root, "pkd_info.csv")
        if osp.exists(pkd_csv):
            self._pkd_info_df = pd.read_csv(pkd_csv, index_col="uuid")
            return self._pkd_info_df
        
        pkd_cols = ["binding_affinity_manual", "binding_affinity_moad", 
                    "binding_affinity_pdbbind", "binding_affinity_bindingdb"]
        has_pkd = self.pl_info_df[pkd_cols].any(axis=1)
        pkd_info_df = self.pl_info_df[has_pkd]
        pkd_info_df = self.add_uuid_col(pkd_info_df)[["uuid"]+pkd_cols].set_index("uuid")
        pkd_info_df = pkd_info_df.applymap(self.parse_verbose_pkd, na_action="ignore")
        pkd_info_df["mean_pkd"] = pkd_info_df.mean(axis=1, skipna=True)
        pkd_info_df = pkd_info_df.dropna(subset=["mean_pkd"])
        pkd_info_df.to_csv(pkd_csv)
        self._pkd_info_df = pkd_info_df
        return self._pkd_info_df

    @staticmethod
    def parse_verbose_pkd(info: Optional[str], single_entry=False) -> Optional[float]:
        # the raw string info is like:
        # EC50=100nM, Ki=583nM, IC50=32nM, Kd=240nM
        # we want to split them and parse individually
        if not single_entry:
            pkd_info: Dict[str, float] = {}
            for entry in info.split(","):
                parsed_entry = BioLipReader.parse_verbose_pkd(entry, True)
                if parsed_entry is None:
                    continue
                pkd_info.update(parsed_entry)
            
            # case one, not pKd available
            if not pkd_info:
                return None
            
            # If both Kd or Ki values are available for the same complex, 
            # the Kd value is chosen as the preferred data.
            if "pKd" in pkd_info:
                return pkd_info["pKd"]
            
            if "pKi" in pkd_info:
                return pkd_info["pKi"]
            
            return None
        
        assert single_entry
        info = info.strip()
        # Consistent with the selection protcol of PDBBind2020 
        # Su, M. et al. J Chem Inf Model 59, 895–913 (2019).
        # Supplimental information, table S2:
        # Estimated binding data, e.g. "Kd ~ 1 nM" or "Ki > 10 μM", are not accepted.
        if ("<" in info) or ("~" in info) or (">" in info) or ("+-" in info):
            return None
        
        if info.startswith("-logKd/Ki="):
            pkd = info.split("-logKd/Ki=")[-1]
            return {"pKd": float(pkd)}
        
        units_mapper = {"pM": 1e-12, "nM": 1e-9, "uM": 1e-6, "mM": 1e-3, "fM": 1e-15}
        def parse_with_unit(ki: str):
            # check fM, pM, nM, uM, mM
            for unit_name in units_mapper:
                if ki.endswith(unit_name):
                    ki = ki.split(unit_name)[0]
                    ki = float(ki) * units_mapper[unit_name]
                    break
            # check unit M
            if isinstance(ki, str) and ki.endswith("M"):
                ki = float(ki.split("M")[0])
            if isinstance(ki, float):
                pkd = -math.log(ki, 10)
                return pkd
            print("Unparse pKd: ", info, ki)
            return None
        
        if info.startswith("Ki="):
            ki = info.split("Ki=")[-1].split()[0]
            return {"pKi": parse_with_unit(ki)}
        if info.startswith("Kd="):
            kd = info.split("Kd=")[-1].split()[0]
            return {"pKd": parse_with_unit(kd)}
        
        # Consistent with the selection protcol of PDBBind2020 
        # Su, M. et al. J Chem Inf Model 59, 895–913 (2019).
        # Supplimental information, table S2:
        # Complexes with known dissociation constants (Kd) or inhibition constants (Ki) are accepted. 
        # Complexes with only half-inhibition or half-effect concentrations (IC50 or EC50) values are not.
        info = info.lower()
        if info.startswith("ic50=") or info.startswith("ec50=") or \
            info.startswith("kon=") or info.startswith("koff=") or \
            info.startswith("ka=") or info.startswith("km="):
            return None
        
        print("Cannot recognize pattern: ", info)
        return None
    
    @property
    def pbind_info_df(self) -> pd.DataFrame:
        if self._pbind_info_df is not None:
            return self._pbind_info_df
        
        pbind_csv = osp.join(self.info_root, "pdbbind_info.csv")
        if osp.exists(pbind_csv):
            self._pbind_info_df = pd.read_csv(pbind_csv)
            return self._pbind_info_df
        
        pbdbind_df = self.to_pbind_df(self.pl_info_df)
        pbdbind_df.to_csv(pbind_csv, index=False)
        self._pbind_info_df = pbdbind_df
        return self._pbind_info_df
    
    def fetch_pdb_info(self, pdb: str) -> None:
        this_df = self.pbind_info_df[self.pbind_info_df["pdb_id"]==pdb]
        for i in range(this_df.shape[0]):
            this_info = this_df.iloc[i]
            prot, lig = self.get_pl(None, this_info)
            print(prot)
            print(lig)
            print("---")
        return
    
    def __len__(self):
        return self.pl_info_df.shape[0]
    
    def get_pl(self, idx: int, info: Union[pd.Series, dict] = None) -> Tuple[str, str]:
        # default protein-ligand pairs
        prot_name, lig_name = self.get_pl_name(idx, info)
        prot = osp.join(self.rec_root, prot_name + ".pdb")
        lig = osp.join(self.lig_root, lig_name + ".pdb")
        return prot, lig
    
    def get_polar_pl(self, idx: int, info: Union[pd.Series, dict] = None) -> Tuple[str, str]:
        # polar hydrogen
        prot_name, lig_name = self.get_pl_name(idx, info)
        prot = osp.join(self.prot_polar_root, prot_name + ".polar.pdb")
        lig = osp.join(self.lig_polar_root, lig_name + ".polar.mol2")
        return prot, lig
    
    def get_polar_pl_uuid(self, uuid: str) -> Tuple[str, str]:
        # polar hydrogen
        prot_name, lig_name = self.break_uuid(uuid)
        prot = osp.join(self.prot_polar_root, prot_name + ".polar.pdb")
        lig = osp.join(self.lig_polar_root, lig_name + ".polar.mol2")
        return prot, lig
    
    def get_lig_pdbqt(self, idx: int, info: Union[pd.Series, dict] = None) -> str:
        __, lig_name = self.get_pl_name(idx, info)
        lig = osp.join(self.lig_pdbqt_root, lig_name + ".pdbqt")
        return lig
    
    def get_linf9_pl(self, idx: int, info: Union[pd.Series, dict] = None) -> Tuple[str, str]:
        # structures after LinF9 optimization
        prot_name, lig_name = self.get_pl_name(idx, info)
        prot = osp.join(self.prot_polar_root, prot_name + ".polar.pdb")
        lig = osp.join(self.lig_linf9_root, lig_name + ".linf9.pdb")
        return prot, lig
    
    def get_linf9_pl_uuid(self, uuid: str) -> Tuple[str, str]:
        # structures after LinF9 optimization
        prot_name, lig_name = self.break_uuid(uuid)
        prot = osp.join(self.prot_polar_root, prot_name + ".polar.pdb")
        lig = osp.join(self.lig_linf9_root, lig_name + ".linf9.pdb")
        return prot, lig
    
    def get_pl_name(self, idx: int, info: Union[pd.Series, dict] = None) -> Tuple[str, str]:
        info = self.pl_info_df.iloc[idx] if info is None else info
        # (name is formed with columns 01,02: i.e., 0102.pdb)
        pdb_id = info["pdb_id"]
        rec_chain = info["receptor_chain"]
        # (name is formed with columns 01,05,06,07: i.e., 01_05_06_07.pdb)
        lig_id_ccd = info["lig_id_ccd"]
        lig_chain = info["lig_chain"]
        lig_serial_num = info["lig_serial_num"]
        prot_name = f"{pdb_id}{rec_chain}"
        lig_name = f"{pdb_id}_{lig_id_ccd}_{lig_chain}_{lig_serial_num}"
        return prot_name, lig_name
    
    def add_uuid_col(self, info_df: pd.DataFrame):
        uuid = []
        for i in tqdm.tqdm(range(info_df.shape[0]), desc="UUID"):
            this_info = info_df.iloc[i]
            uuid.append(self.get_uuid(this_info))
        info_df["uuid"] = uuid
        return info_df
    
    def get_uuid(self, info: Union[pd.Series, dict]) -> str:
        prot_name, lig_name = self.get_pl_name(None, info)
        return f"UUID_{prot_name}__{lig_name}"
    
    def break_uuid(self, uuid: str) -> Tuple[str, str]:
        prot_name, lig_name = uuid.split("__")
        prot_name = prot_name.split("_")[-1]
        return prot_name, lig_name
    
    @property
    def stats(self):
        if self._stats is not None:
            return self._stats
        
        stats = {}
        stats["n_total"] = self.pl_info_df.shape[0]
        stats["n_pdbbind"] = self.pbind_info_df.shape[0]
        stats["n_non_cov"] = self.pl_lvl_1_info_df.shape[0]
        stats["n_non_cov_pbind"] = self.to_pbind_df(self.pl_lvl_1_info_df).shape[0]
        stats["n_lvl2_linf9"] = self.pl_lvl_2_info_df.shape[0]
        stats["n_lvl2_linf9_pbind"] = self.to_pbind_df(self.pl_lvl_2_info_df).shape[0]
        stats["n_lvl3_linf9"] = self.pl_lvl_3_info_df.shape[0]
        stats["n_lvl3_linf9_pbind"] = self.to_pbind_df(self.pl_lvl_3_info_df).shape[0]
        stats["n_lvl4_linf9"] = self.pl_lvl_4_info_df.shape[0]
        stats["n_lvl4_linf9_pbind"] = self.to_pbind_df(self.pl_lvl_4_info_df).shape[0]
        stats["n_final"] = self.pl_lvl_final_info_df.shape[0]
        stats["n_final_pbind"] = self.to_pbind_df(self.pl_lvl_final_info_df).shape[0]

        self._stats = stats
        return self._stats
    
    def vis_resolution(self):
        fig, ax = plt.subplots(1, 1)
        sns.histplot(self.pl_info_df, x="resolution")
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_root, "resolution_dist.png"))
        plt.close()

        fig, ax = plt.subplots(1, 1)
        sns.histplot(self.pl_info_df, x="resolution")
        ax.set_xlim([0, 10])
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_root, "resolution_0_10_dist.png"))
        plt.close()

def test_reader():
    reader = BioLipReader("/vast/sx801/geometries/BioLiP_updated_set")

    print(reader.uuid2pkd("UUID_9hvpB__9hvp_0E9_A_1"))

if __name__ == "__main__":
    test_reader()
