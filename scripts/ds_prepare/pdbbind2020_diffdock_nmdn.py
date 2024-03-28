import argparse
from glob import glob
import os.path as osp
import tempfile
from tqdm import tqdm
import numpy as np

from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.pdb2020_ds_reader import PDB2020DSReader
from geometry_processors.process.mp_pyg_runner import ArrayPygRunner, proc_hetero_graph
from geometry_processors.process.proc_hydrogen import LigPolarConverter

reader = PDB2020DSReader("/PDBBind2020_OG")
DS_NAME = "PBind2020OG.diffdock-nmdn.hetero.polar.polar"

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
parser.add_argument("--run_collate", action="store_true")
args = parser.parse_args()
array_id: int = args.array_id

def run_array():
    pdb_folders = glob(f"/vast/sx801/geometries/PDBBind2020_DiffDock-sampled/raw_predicts/????")
    pdb_ids = [osp.basename(pdb_folder) for pdb_folder in pdb_folders]
    pdb_ids.sort()
    pdb_ids = np.array_split(pdb_ids, 100)[array_id]

    info_list = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for pdb_id in tqdm(pdb_ids):
            prot_polarh = reader.pdb2polar_prot(pdb_id)
            for raw_lig in glob(f"/vast/sx801/geometries/PDBBind2020_DiffDock-sampled/raw_predicts/{pdb_id}/rank*_confidence*.sdf"):
                fname = osp.basename(raw_lig)
                polarh_fname = f"{pdb_id}.{fname}"
                polarh_file = osp.join(tmpdirname, polarh_fname)
                LigPolarConverter(raw_lig, polarh_file).run()

                file_handle = polarh_fname.split(".sdf")[0]
                info_list.append(MPInfo(protein_pdb=prot_polarh, ligand_sdf=polarh_file, pdb=pdb_id, 
                                        file_handle=file_handle, pkd=reader.pdb2pkd[pdb_id]))
                
        pyg_runner = ArrayPygRunner(array_id, info_list=info_list, proc_fn=proc_hetero_graph, 
                                    ds_name=DS_NAME)
        pyg_runner.run_array()


def run_collate():
    pyg_runner = ArrayPygRunner(None, info_list=[], proc_fn=proc_hetero_graph, 
                                ds_name=DS_NAME)
    pyg_runner.run_collate(nparts=10)

if args.run_collate:
    run_collate()
else:
    run_array()

