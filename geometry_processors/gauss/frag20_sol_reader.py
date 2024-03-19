from collections import defaultdict
import os
import os.path as osp
from glob import glob
from typing import List, Tuple
from torch_geometric.data import Data
from rdkit.Chem import MolFromMol2Block, MolFromMol2File

import pandas as pd
import numpy as np
from geometry_processors.gauss.read_gauss_log import Gauss16Log
from geometry_processors.lazy_property import lazy_property

from geometry_processors.pl_dataset.csv2input_list import MPInfo

from ase.units import Hartree, eV
hartree2ev = Hartree / eV

class Frag20SolvReader:
    def __init__(self, droot: str) -> None:
        # Frag20-solv-678k
        self.droot = droot

    def gather_sdf(self, info: pd.Series, **kwargs) -> str:
        source = info["SourceFile"]
        if source == "Conf20": return self.gather_conf20_sdf(info["FileHandle"], **kwargs)
        return self.gather_frag20_csd20_sdf(info["FileHandle"], **kwargs)

    def gather_conf20_sdf(self, file_handle: str, geom="mmff", polarh=False):
        source, sample_id = file_handle.split("_")
        assert source == "Conf20", source
        assert geom in ["mmff", "qm"], geom
        folder_name: str = "mmff_sdfs" if geom == "mmff" else "qm_sdf"
        polarh: str = ".polarh" if polarh else ""
        file_name: str = f"{sample_id}.sdf" if geom == "mmff" else f"{sample_id}.qm.sdf"
        return osp.join(self.droot, f"Conf20_sol{polarh}", folder_name, file_name)
    
    def gather_frag20_csd20_sdf(self, file_handle: str, geom="mmff", polarh=False):
        source, sample_id = file_handle.split("_")
        polarh: str = ".polarh" if polarh else ""

        extra = ".opt" if geom == "qm" else ""

        try:
            source = int(source)
            this_sdf = f"{self.droot}/Frag20_sdfs{polarh}/Frag20_{source}_data/{sample_id}{extra}.sdf"
        except ValueError:
            assert source in ["PubChem", "Zinc", "CCDC"]
            if source in ["PubChem", "Zinc"]:
                this_sdf = f"{self.droot}/Frag20_sdfs{polarh}/Frag20_9_data/{source.lower()}/{sample_id}{extra}.sdf"
            else:
                extra = ".opt" if geom == "qm" else "_min"
                this_sdf = f"{self.droot}/CSD20_sdfs{polarh}/cry_min/{sample_id}{extra}.sdf"
        return this_sdf

    @lazy_property
    def info_df(self) -> pd.DataFrame:
        frag20_csd20_csv: str = osp.join(self.droot, "frag20_solvation_with_fl.csv")
        frag20_csd20_df: pd.DataFrame = pd.read_csv(frag20_csd20_csv)
        interested = ["SourceFile", "FileHandle", "gasEnergy(au)", "watEnergy(au)", "octEnergy(au)",
                    "CalcSol", "CalcOct", "watOct"]
        frag20_csd20_df = frag20_csd20_df[interested]

        conf20_csv: str = osp.join(self.droot, "Conf20_sol", "summary", "gauss_sol.csv")
        conf20_df: pd.DataFrame = pd.read_csv(conf20_csv)
        def _conf20_fl(sample_id: int):
            return f"Conf20_{sample_id}"
        conf20_df["FileHandle"] = conf20_df["sample_id"].map(_conf20_fl)
        interested = ["FileHandle", "water_E_atom(eV)", "gas_E_atom(eV)", "1-octanol_E_atom(eV)", 
                      "water_gas(kcal/mol)", "1-octanol_gas(kcal/mol)", "water_1-octanol(kcal/mol)"]
        conf20_df = conf20_df[interested]
        rename = {"water_gas(kcal/mol)": "CalcSol", "1-octanol_gas(kcal/mol)": "CalcOct",
                  "water_1-octanol(kcal/mol)": "watOct"}
        conf20_df = conf20_df.rename(rename, axis=1)
        conf20_df["SourceFile"] = ["Conf20"] * conf20_df.shape[0]

        return pd.concat([frag20_csd20_df, conf20_df], join="outer").reset_index()
    
    def process_single_entry(self, info: pd.Series, **kwargs) -> Tuple[str, dict]:
        sdf_file: str = self.gather_sdf(info, **kwargs)
        log_reader: Gauss16Log = Gauss16Log(None, log_sdf=sdf_file, supress_warning=True)
        
        # inject reference so we don't load the refernce file everytime we are processing a new molecule
        log_reader._reference = self.reference
        geom_dict: dict = log_reader.get_basic_dict()
        geom_dict.update(self.compute_props(info, **kwargs))

        return Data(**geom_dict)
    
    def compute_props(self, info: pd.Series, **kwargs) -> dict:
        # gather energetics terms
        # computed reference energy if needed
        # the reference energy is computed using all hydrogens
        kwargs["polarh"] = False
        sdf_file_allh: str = self.gather_sdf(info, **kwargs)
        log_reader_allh: Gauss16Log = Gauss16Log(None, log_sdf=sdf_file_allh, supress_warning=True)
        log_reader_allh._reference = self.reference

        prop_dict = {}
        gasEnergy, watEnergy, octEnergy = info["gas_E_atom(eV)"], info["water_E_atom(eV)"], info["1-octanol_E_atom(eV)"]
        if not np.isfinite(gasEnergy) or gasEnergy is None:
            gasEnergy, watEnergy, octEnergy = info["gasEnergy(au)"], info["watEnergy(au)"], info["octEnergy(au)"]
            ref = log_reader_allh.reference_u0
            gasEnergy = hartree2ev * gasEnergy - ref
            watEnergy = hartree2ev * watEnergy - ref
            octEnergy = hartree2ev * octEnergy - ref
        prop_dict["gasEnergy"] = gasEnergy
        prop_dict["watEnergy"] = watEnergy
        prop_dict["octEnergy"] = octEnergy
        for key in ["CalcSol", "CalcOct", "watOct", "FileHandle", "SourceFile"]:
            prop_dict[key] = info[key]
        return prop_dict
    
    @lazy_property
    def reference(self):
        dummy_gauss_reader = Gauss16Log(None, None)
        return dummy_gauss_reader.reference

if __name__ == "__main__":
    reader = Frag20SolvReader("/vast/sx801/geometries/Frag20-sol")
    print(reader.info_df)
