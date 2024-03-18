"""
Read protein-peptide interaction dataset
"""
from collections import defaultdict
import os
import os.path as osp
from glob import glob
from typing import Dict
from prody import parsePDB
from tqdm import tqdm
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import json


class ProtPepDSReader:
    def __init__(self, ds_root) -> None:
        self.ds_root = ds_root
        # Reference crystal structures
        self.ref_root = osp.join(ds_root, "PDB_ref")
        # Structures generated using AlphaFold2
        self.af2_root = osp.join(ds_root, "collected_AF_models")
        self.dockq_csv = osp.join(ds_root, "label_n_split", "song_biolip_2369_full.csv")

        self._ref_pdbs = None
        self._af2_pdbs = None
        self._info_df = None
        self._total_pdbs = None
        self._pdb_by_split = None
        self._dockq_df = None
        self._dockq_query = None
        self._dockq_query_df = None

    @property
    def total_pdbs(self):
        if self._total_pdbs is not None:
            return self._total_pdbs
        
        __ = self.pdb_by_split
        return self._total_pdbs

    @property
    def pdb_by_split(self):
        if self._pdb_by_split is not None:
            return self._pdb_by_split
        
        self._total_pdbs = list(set(self.info_df["pdb_id"].values.tolist()))
        pdb_by_split = defaultdict(lambda: [])
        for info_pdb in self._total_pdbs:
            this_df = self.info_df[self.info_df["pdb_id"]==info_pdb]
            this_split = this_df["data_class"].iloc[0]
            # the same pdb should be in the same split.
            assert np.sum(this_df["data_class"] == this_split) == this_df.shape[0], this_df
            pdb_by_split[this_split].append(info_pdb)
        self._pdb_by_split = pdb_by_split
        return self._pdb_by_split

    @property
    def info_df(self):
        if self._info_df is not None:
            return self._info_df
        
        info_csv = osp.join(self.ds_root, "label_n_split", "song_biolip_2369_full.csv")
        info_df = pd.read_csv(info_csv)
        self._info_df = info_df
        return self._info_df
    
    # the provided AF2 generated protein structures contain ALL hydrogens
    def get_af2_prots(self, pdb: str):
        res = glob(osp.join(self.ds_root, "collected_AF_models", pdb, "ranked_*_sp_pro.pdb"))
        res.sort(key=lambda s: int(osp.basename(s).split("_")[1]))
        return res

    # the provided AF2 generated peptide structures contain ALL hydrogens
    def get_af2_peptides(self, pdb: str):
        res = glob(osp.join(self.ds_root, "collected_AF_models", pdb, "ranked_*_sp_pep.pdb"))
        res.sort(key=lambda s: int(osp.basename(s).split("_")[1]))
        return res
    
    def get_af2_polarH_peptides(self, pdb: str):
        res = [p.replace("collected_AF_models", "collected_AF_polarH_models") 
               for p in self.get_af2_peptides(pdb)]
        return res
    
    def get_af2_model_id(self, af2_name: str):
        return int(osp.basename(af2_name).split("_")[1])
    
    # the provided crystal protein structures contain NO hydrogen
    def get_ref_prot(self, pdb: str):
        return osp.join(self.ds_root, "PDB_ref", pdb, "protein.pdb")
    
    def get_ref_renumH_prot(self, pdb: str):
        return osp.join(self.ds_root, "PDB_ref_renumH", pdb, "protein.pdb")
    
    # the provided crystal peptide structures contain NO hydrogen
    def get_ref_peptide(self, pdb: str):
        return osp.join(self.ds_root, "PDB_ref", pdb, f"{pdb}_ref-pep.pdb")
    
    def get_ref_polarH_peptide(self, pdb: str):
        return osp.join(self.ds_root, "PDB_ref_polarH", pdb, f"{pdb}_ref-pep.pdb")

    @property
    def dockq_df(self):
        if self._dockq_df is not None:
            return self._dockq_df
        
        self._dockq_df = pd.read_csv(self.dockq_csv)
        return self._dockq_df
    
    @property
    def dockq_query(self):
        if self._dockq_query is not None:
            return self._dockq_query
        
        dockq_query = {}
        for i in range(self.dockq_df.shape[0]):
            this_df = self.dockq_df.iloc[i]
            pdb_id = this_df["pdb_id"]
            model_id = this_df["af_model_id"]
            dockq = this_df["pdb2sql_DockQ"]
            dockq_query[(pdb_id, model_id)] = dockq

        dockq_query_df = defaultdict(lambda: [])
        for key in dockq_query.keys():
            dockq_query_df["pdb"].append(key[0])
            dockq_query_df["model_id"].append(key[1])
            dockq_query_df["dockq"].append(dockq_query[key])
        dockq_query_df = pd.DataFrame(dockq_query_df).set_index(["pdb", "model_id"])
        self._dockq_query_df = dockq_query_df
        self._dockq_query = dockq_query
        return self._dockq_query

    @property
    def ref_pdbs(self):
        raise ValueError("Use self.total_pdbs")
        if self._ref_pdbs is not None:
            return self._ref_pdbs
        
        ref_pdbs = glob(osp.join(self.ref_root, "????"))
        self._ref_pdbs = ref_pdbs
        return self._ref_pdbs
    
    @property
    def af2_pdbs(self):
        raise ValueError("Use self.total_pdbs")
        if self._af2_pdbs is not None:
            return self._af2_pdbs
        
        af2_pdbs = glob(osp.join(self.af2_root, "????"))
        self._af2_pdbs = af2_pdbs
        return self._af2_pdbs
    
class ProtPepDSStat(ProtPepDSReader):
    def __init__(self, ds_root, save_root) -> None:
        super().__init__(ds_root)
        os.makedirs(save_root, exist_ok=True)
        self.save_root = save_root

        # disable the logging from prody
        logger = logging.getLogger(".prody")
        logger.setLevel(logging.CRITICAL)

        # setup logger
        logging.basicConfig(filename=osp.join(save_root, "ds_stat.log"), format="%(asctime)s %(message)s", filemode="w")
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # More detailed stats of the dataset: sequence, num protein and peptide residues.
        self._detailed_df = None
        self.detailed_csv = osp.join(ds_root, "label_n_split", "detailed_info.csv")
        self._ref_renumH_seq_df = None
        self.ref_renumH_seq_csv = osp.join(ds_root, "label_n_split", "ref_renumH_seq.csv")

    @property
    def ref_renumH_seq_df(self) -> pd.DataFrame:
        if self._ref_renumH_seq_df is not None:
            return self._ref_renumH_seq_df
        
        if osp.exists(self.ref_renumH_seq_csv):
            self._ref_renumH_seq_df = pd.read_csv(self.ref_renumH_seq_csv)
            return self._ref_renumH_seq_df
        
        from geometry_processors.pl_dataset.prot_utils import pdb2seq
        info = defaultdict(lambda: [])
        for pdb in tqdm(self.total_pdbs):
            ref_renumH_prot = self.get_ref_renumH_prot(pdb)
            seq = pdb2seq(ref_renumH_prot)
            info["pdb"].append(pdb)
            info["seq"].append(seq)
        self._ref_renumH_seq_df = pd.DataFrame(info)
        self._ref_renumH_seq_df.to_csv(self.ref_renumH_seq_csv)
        return self._ref_renumH_seq_df
    
    @property
    def ref_renumH_seq_info(self) -> Dict[str, str]:
        info_series = self.ref_renumH_seq_df.set_index("pdb")["seq"]
        return info_series.to_dict()

    @property
    def detailed_df(self):    
        if self._detailed_df is not None:
            return self._detailed_df
        
        if osp.exists(self.detailed_csv):
            self._detailed_df = pd.read_csv(self.detailed_csv)
            return self._detailed_df
        
        from geometry_processors.pl_dataset.prot_utils import pdb2seq
        detailed_pdb_info = defaultdict(lambda: [])
        ckpt_json = osp.join(self.ds_root, "label_n_split", "detailed_info.ckpt.json")
            
        for i, pdb in tqdm(enumerate(self.total_pdbs), total=len(self.total_pdbs)):
            consistent_af2 = True

            ref_prot = self.get_ref_prot(pdb)
            ref_prot_seq = pdb2seq(ref_prot)
            af2_prot_seq = None
            for af2_prot in self.get_af2_prots(pdb):
                this_af2_prot_seq = pdb2seq(af2_prot)
                if af2_prot_seq is None:
                    af2_prot_seq = this_af2_prot_seq
                    continue

                if af2_prot_seq != this_af2_prot_seq:
                    consistent_af2 = False
                    self.logger.warn("----------ERROR_PROTEIN-------------")
                    self.logger.warn(f"Inconsistent found at {pdb}")
                    self.logger.warn(f"Prev Prot seq: {af2_prot_seq}")
                    self.logger.warn(f"This Prot seq: {this_af2_prot_seq}")
                    self.logger.warn(f"Inconsistent AF2 File: {af2_prot}")
                    break
            if not consistent_af2:
                continue

            ref_peptide = self.get_ref_peptide(pdb)
            ref_peptide_seq = pdb2seq(ref_peptide)
            af2_peptide_seq = None
            for af2_peptide in self.get_af2_peptides(pdb):
                this_af2_peptide_seq = pdb2seq(af2_peptide)
                if af2_peptide_seq is None:
                    af2_peptide_seq = this_af2_peptide_seq
                    continue

                if af2_peptide_seq != this_af2_peptide_seq:
                    consistent_af2 = False
                    self.logger.warn("----------ERROR_PEPTIDE-------------")
                    self.logger.warn(f"Inconsistent found at {pdb}")
                    self.logger.warn(f"Prev Peptide seq: {af2_peptide_seq}")
                    self.logger.warn(f"This Peptide seq: {this_af2_peptide_seq}")
                    self.logger.warn(f"Inconsistent AF2 File: {af2_peptide}")
                    break
            if not consistent_af2:
                continue

            detailed_pdb_info["pdb"].append(pdb)
            prot_ag = parsePDB(ref_prot)
            detailed_pdb_info["#ProtAtoms"].append(prot_ag.numAtoms())
            detailed_pdb_info["#ProtResidues"].append(len(ref_prot_seq))
            peptide_ag = parsePDB(ref_peptide)
            detailed_pdb_info["#PeptideAtoms"].append(peptide_ag.numAtoms())
            detailed_pdb_info["#PeptideResidues"].append(len(ref_peptide_seq))
            detailed_pdb_info["prot_ref_consistent_af2"].append(ref_prot_seq == af2_prot_seq)
            detailed_pdb_info["peptide_ref_consistent_af2"].append(ref_peptide_seq == af2_peptide_seq)
            detailed_pdb_info["ref_prot_seq"].append(ref_prot_seq)
            detailed_pdb_info["af2_prot_seq"].append(af2_prot_seq)
            detailed_pdb_info["ref_peptide_seq"].append(ref_peptide_seq)
            detailed_pdb_info["af2_peptide_seq"].append(af2_peptide_seq)

            if (i+1) % 100 == 0:
                with open(ckpt_json, "w") as f:
                    json.dump(detailed_pdb_info, f)

        detailed_df = pd.DataFrame(detailed_pdb_info)
        detailed_df.to_csv(self.detailed_csv, index=False)
        self._detailed_df = detailed_df
        return self._detailed_df
    
    def ds_stat(self):
        __ = self.detailed_df

        self.draw_dockq_list()
        self.draw_aa_atom_dist()

        for split in self.pdb_by_split:
            self.logger.info(f"{split} PDBs: {len(self.pdb_by_split[split])}")

    def draw_dockq_list(self):
        import seaborn as sns
        savepic = osp.join(self.save_root, "dockq_dist.png")
        if osp.exists(savepic):
            return
        
        info_pdbs = set(self.info_df["pdb_id"].values.tolist())
        self.logger.info(f"Total number of different pdbs in the info_csv: {len(info_pdbs)}")
        min_dockq = []
        median_dockq = []
        max_dockq = []
        pdb_by_split = defaultdict(lambda: [])
        for info_pdb in info_pdbs:
            this_df = self.info_df[self.info_df["pdb_id"]==info_pdb]
            dockq = this_df["pdb2sql_DockQ"].values
            min_dockq.append(np.min(dockq).item())
            median_dockq.append(np.median(dockq).item())
            max_dockq.append(np.max(dockq).item())

            this_split = this_df["data_class"].iloc[0]
            # the same pdb should be in the same split.
            assert np.sum(this_df["data_class"] == this_split) == this_df.shape[0], this_df
            pdb_by_split[this_split].append(pdb_by_split)
        sns_df = {"Max DockQ": max_dockq, "Median DockQ": median_dockq, "Min DockQ": min_dockq}
        sns_df = pd.DataFrame(sns_df)
        fig, ax = plt.subplots(1, 1)
        sns.histplot(sns_df)
        plt.savefig(savepic)
        plt.close()

    def draw_aa_atom_dist(self):
        savepic = osp.join(self.save_root, "aa_atom_dist.png")
        savepic_pep = osp.join(self.save_root, "aa_atom_peptide_dist.png")
        if osp.exists(savepic) and osp.exists(savepic_pep):
            return
        
        fig, ax = plt.subplots(1, 1)
        ax.set(xscale="log")
        wanted_cols = ["#PeptideAtoms","#PeptideResidues", "#ProtAtoms","#ProtResidues"]
        sns.histplot(self.detailed_df[wanted_cols])
        plt.savefig(savepic)
        plt.close()

        fig, ax = plt.subplots(1, 1)
        wanted_cols = ["#PeptideAtoms","#PeptideResidues"]
        sns.histplot(self.detailed_df[wanted_cols])
        plt.savefig(savepic_pep)
        plt.close()

