from glob import glob
import os.path as osp
import pandas as pd
"""
Read protein and ligand information from disk for Chao Yang's PL dataset.
"""

class PLReader:
    def __init__(self, ds_root) -> None:
        self.ds_root = ds_root
        self.label_csvs = glob(osp.join(self.ds_root, "label", "*", "*.csv"))

        self._prot_templ_query = {}
        self._lig_templ_query = {}
        self._pdb_key_query = {}
        self._lig_folder_query = {}

        self._len = None

    def iter_pdb_info(self):
        for csv in self.label_csvs:
            source = osp.basename(csv).split(".csv")[0]
            split = osp.basename(osp.dirname(csv))
            label_df = pd.read_csv(csv)
            for i in range(label_df.shape[0]):
                this_info = label_df.iloc[i].to_dict()
                this_info["source"] = source
                this_info["split"] = split
                pdb_key = self._get_pdb_key(source)
                this_info["pdb"] = this_info[pdb_key]
                yield this_info

    def info2lig_polar_sdf(self, info: dict):
        lig_attr = ".polar"
        attr_folder = "structure_polarH"
        return self._info2lig_sdf(info, lig_attr, attr_folder)
    
    def info2prot_renum_pdb(self, info: dict):
        pro_attr = ".renum"
        attr_folder = "structure_fixed"
        return self._info2prot_pdb(info, pro_attr, attr_folder)
    
    def info2prot_martini_pdb(self, info: dict):
        pro_attr = ".martini"
        attr_folder = "structure_martini"
        return self._info2prot_pdb(info, pro_attr, attr_folder)
    
    def __len__(self):
        if self._len is not None:
            return self._len
        
        _len = sum([pd.read_csv(csv).shape[0] for csv in self.label_csvs])
        self._len = _len
        return self._len
    
    def _info2prot_pdb(self, info: dict, pro_attr, attr_folder):
        source = info["source"]
        split = info["split"]
        pdb = info["pdb"]
        templ = {"pdb": pdb, "pro_attr": pro_attr}
        if source.startswith("CSAR_decoy_"):
            templ["o_index"] = info["o_index"]
        elif source.startswith("binder4_"):
            templ["o_index"] = "{:02d}".format(info["o_index"])
        elif source.startswith("binder5_"):
            templ["o_index"] = "{:02d}".format(info["o_index"])
            templ["ligand_id"] = "{:05d}".format(info["ligand_id"])
        prot_templ = self._get_prot_templ(source)
        protein_pdb = osp.join(self.ds_root, attr_folder, split, source, "protein", prot_templ.format(**templ))
        return protein_pdb

    def _info2lig_sdf(self, info: dict, lig_attr, attr_folder):
        source = info["source"]
        split = info["split"]
        pdb = info["pdb"]
        templ = {"pdb": pdb, "lig_attr": lig_attr}
        if source.startswith("CSAR_decoy_"):
            templ["o_index"] = info["o_index"]
        elif source.startswith("binder4_"):
            templ["o_index"] = "{:02d}".format(info["o_index"])
        elif source.startswith("binder5_"):
            templ["o_index"] = "{:02d}".format(info["o_index"])
            templ["ligand_id"] = "{:05d}".format(info["ligand_id"])
        lig_folder = self._get_lig_folder(source)
        lig_templ = self._get_lig_templ(source)
        ligand_sdf = osp.join(self.ds_root, attr_folder, split, source, lig_folder, lig_templ.format(**templ))
        return ligand_sdf

    def _get_prot_templ(self, source: str):
        if source in self._prot_templ_query:
            return self._prot_templ_query[source]
        
        prot_templ = "{pdb}_protein{pro_attr}.pdb"
        if source == "PDBbind_refined_wat":
            prot_templ = "{pdb}_protein_RW{pro_attr}.pdb"
        elif source.startswith("CSAR_dry"):
            prot_templ = "{pdb}{pro_attr}.pdb"
        elif source.startswith("CSAR_decoy_"):
            prot_templ = "{pdb}{pro_attr}.pdb"
        self._prot_templ_query[source] = prot_templ
        return prot_templ
    
    def _get_lig_templ(self, source: str):
        if source in self._lig_templ_query:
            return self._lig_templ_query[source]
        
        lig_templ = "{pdb}_ligand{lig_attr}.sdf"
        if source.startswith("E2E") or source.startswith("val_E2E"):
            lig_templ = "{pdb}{lig_attr}.sdf"
        elif source.startswith("binder4_"):
            lig_templ = "{pdb}_docked_{o_index}{lig_attr}.sdf"
        elif source.startswith("CSAR_decoy_"):
            lig_templ = "{pdb}_decoys_{o_index}{lig_attr}.sdf"
        elif source.startswith("binder5_"):
            lig_templ = "{ligand_id}_{pdb}_{o_index}{lig_attr}.sdf"
        self._lig_templ_query[source] = lig_templ
        return lig_templ
    
    def _get_pdb_key(self, source: str):
        if source in self._pdb_key_query:
            return self._pdb_key_query[source]
        pdb_key = "pdb"
        if source.startswith("binder5_"):
            pdb_key = "refPDB"
        self._pdb_key_query[source] = pdb_key
        return pdb_key
    
    def _get_lig_folder(self, source: str):
        if source in self._lig_folder_query:
            return self._lig_folder_query[source]
        lig_folder = "pose"
        if source.startswith("E2E") or source.startswith("val_E2E"):
            lig_folder = "docked_pose"
        self._lig_folder_query[source] = lig_folder
        return lig_folder

