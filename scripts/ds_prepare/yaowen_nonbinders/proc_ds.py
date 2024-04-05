import argparse
import os.path as osp
import tempfile

from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import pandas as pd

from geometry_processors.pl_dataset.casf2016_blind_docking import CASF2016BlindDocking
from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
from geometry_processors.pl_dataset.yaowen_nonbinder_reader import NonbinderReader
from geometry_processors.process.mp_pyg_runner import ArrayPygRunner, DataModifier, proc_hetero_graph
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.process.proc_hydrogen import LigPolarConverter, ProtPolarConverter


def prot_polarh():
    reader = NonbinderReader()
    for uniprot_id in tqdm(reader.uniprot_ids):
        polarh = reader.uniprot_id2prot_polarh(uniprot_id)
        if osp.exists(polarh):
            continue
        og_prot = reader.uniprot_id2prot_og(uniprot_id)
        conv = ProtPolarConverter(og_prot, polarh)
        conv.run()


def proc():
    parser = argparse.ArgumentParser()
    parser.add_argument("--array_id", type=int)
    args = parser.parse_args()
    array_id: int = args.array_id
    
    mp_info = []
    reader = NonbinderReader()
    split_chunks = np.array_split(np.arange(reader.info_df.shape[0]), 100)
    tmpdirname = "/vast/sx801/geometries/Yaowen_nonbinders/pose_diffdock/polarh"
    for i in split_chunks[array_id]:
        this_info = reader.info_df.iloc[i]
        uniprot_id = this_info["Uniprot_ID"]
        file_handle = this_info["file_handle"]
        polar_prot = reader.uniprot_id2prot_polarh(uniprot_id)
        for raw_lig in reader.fl2ligs(file_handle):
            fname = osp.basename(raw_lig)
            rank:str = fname.split("_")[0]
            file_handle_ranked = f"{file_handle}.src{rank}"
            polarh_fname = f"{file_handle_ranked}.sdf"
            polar_lig = osp.join(tmpdirname, polarh_fname)
            LigPolarConverter(raw_lig, polar_lig).run()
            this_info = {"protein_pdb": polar_prot, "ligand_sdf": polar_lig, 
                        "file_handle": file_handle_ranked, "pdb": uniprot_id}
            mp_info.append(MPInfo(**this_info))
    pyg_runner = ArrayPygRunner(array_id, info_list=mp_info, proc_fn=proc_hetero_graph, 
                                ds_name="yaowen-nonbinders-hetero")
    pyg_runner.run_array()
    # pyg_runner.run_collate(nparts=10)

def collate():
    pyg_runner = ArrayPygRunner(0, info_list=[], proc_fn=proc_hetero_graph, 
                                ds_name="yaowen-nonbinders-hetero", data_modifier=Modifier())
    pyg_runner.run_collate(nparts=20)

class Modifier(DataModifier):
    def __init__(self) -> None:
        super().__init__()
        self.pcba_entries: pd.DataFrame = pd.read_csv("/scratch/sx801/scripts/DiffDock-NMDN/scripts/ds_prepare/yaowen_nonbinders/lit_pcba_entries.tsv", sep="\t")
        self.pcba_uniprot_ids = set(self.pcba_entries["Entry"].values.reshape(-1).tolist())
        self.nonbinder_reader = NonbinderReader()
        self.info_df = self.nonbinder_reader.info_df.set_index("file_handle")

    def __call__(self, data: Data) -> Data:
        if data.pdb in self.pcba_uniprot_ids:
            return None
        fl: str = data.file_handle
        fl = ".".join(fl.split(".")[:-1])
        data.pkd = self.info_df.loc[fl, "Activity"].item()

        return data

if __name__ == "__main__":
    collate()
