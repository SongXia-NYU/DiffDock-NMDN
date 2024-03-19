from typing import Dict, List
import pandas as pd
import os.path as osp
import yaml
from tqdm import tqdm
from glob import glob
import rdkit
from rdkit.Chem import MolFromSmiles, MolToInchi, SDMolSupplier, RemoveStereochemistry

from geometry_processors.lazy_property import lazy_property
from geometry_processors.misc import cached_dict


class QM9Reader:
    def __init__(self, droot: str) -> None:
        self.droot: str = droot
        self.info_root: str = osp.join(droot, "info")
        self.geometry_root: str = osp.join(droot, "geometries")
        self.qm_root: str = osp.join(self.geometry_root, "qm")
        self.mmff_root: str = osp.join(self.geometry_root, "mmff")

    @lazy_property
    def molids(self) -> List[str]:
        return list(self.molid2jlid.keys())

    def molid2mmff_sdf(self, molid: str) -> str:
        jlid = self.molid2jlid[molid]
        return osp.join(self.mmff_root, f"{jlid}.mmff.sdf")
    
    def molid2qm_sdf(self, molid: str) -> str:
        jlid = self.molid2jlid[molid]
        return osp.join(self.qm_root, f"{jlid}.sdf")
    
    def molid2props(self, molid: str) -> dict:
        info = self.label_df.loc[molid]
        info_dict = info.to_dict()
        info_dict["mol_id"] = molid
        return info_dict
    
    @lazy_property
    def label_df(self) -> pd.DataFrame:
        return pd.read_csv(osp.join(self.droot, "qm9.csv")).set_index("mol_id")

    @lazy_property
    def molid2jlid(self) -> Dict[str, int]:
        molid2jlid_yaml: str = osp.join(self.info_root, "molid2jlid.yaml")
        if osp.exists(molid2jlid_yaml):
            with open(molid2jlid_yaml) as f:
                return yaml.safe_load(f)
        
        res: Dict[str, int] = {}
        inchi2jlid: Dict[str, int] = {}
        for jlid in self.jlid2inchi:
            inchi2jlid[self.jlid2inchi[jlid]] = jlid
        
        for molid in self.molid2inchi:
            inchi = self.molid2inchi[molid]
            if inchi not in inchi2jlid:
                continue
            res[molid] = inchi2jlid[inchi]
        with open(molid2jlid_yaml, "w") as f:
            yaml.safe_dump(res, f)
        return res

    @lazy_property
    def molid2inchi(self) -> Dict[str, str]:
        molid2inchi_yaml: str = osp.join(self.info_root, "molid2inchi.yaml")
        if osp.exists(molid2inchi_yaml):
            with open(molid2inchi_yaml) as f:
                return yaml.safe_load(f)

        res: Dict[str, str] = {}
        for i in tqdm(range(self.label_df.shape[0])):
            this_entry: pd.Series = self.label_df.iloc[i]
            molid: str = this_entry.index
            smiles: str = this_entry["smiles"]
            inchi = MolToInchi(MolFromSmiles(smiles))
            res[molid] = inchi
        with open(molid2inchi_yaml, "w") as f:
            yaml.safe_dump(res, f)
        return res
    
    @lazy_property
    def jlid2inchi(self) -> Dict[int, str]:
        """
        Mol ID assigned by Jianing Lu.
        """
        jlid2inchi_yaml: str = osp.join(self.info_root, "jlid2inchi.yaml")
        if osp.exists(jlid2inchi_yaml):
            with open(jlid2inchi_yaml) as f:
                return yaml.safe_load(f)
            
        mmff_sdfs = glob(osp.join(self.mmff_root, "*.mmff.sdf"))
        res: Dict[int, str] = {}
        for mmff_sdf in tqdm(mmff_sdfs):
            jlid = osp.basename(mmff_sdf).split(".mmff.sdf")[0]
            jlid = int(jlid)
            with SDMolSupplier(mmff_sdf) as suppl:
                mol = suppl[0]
            mol.RemoveAllConformers()
            RemoveStereochemistry(mol)
            inchi = MolToInchi(mol)
            res[jlid] = inchi
        with open(jlid2inchi_yaml, "w") as f:
            yaml.safe_dump(res, f)
        return res


