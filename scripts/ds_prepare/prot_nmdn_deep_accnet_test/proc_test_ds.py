from glob import glob
import os
import os.path as osp

from typing import List

from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.process.mp_pyg_runner import MultiProcessPygRunner, proc_single_protein_implicit_mindist


pdb_files = glob(f"/scratch/xd638/20240407_DeepAccNet_unzip/test/new_pdbs_polarH/*/*.pdb")


# -----------------You need to modify the following variables----------------- #

# this is the name of the data set
DS_NAME = "DeepAccNet_testset"
# -----------------------------------------------------------------------------#


info_list: List[MPInfo] = []
for pdb_file in pdb_files:
    pdb_id = osp.basename(osp.dirname(pdb_file))
    this_info = MPInfo(protein_pdb=pdb_file, pdb=pdb_id)
    info_list.append(this_info)

runner = MultiProcessPygRunner(info_list=info_list, proc_fn=proc_single_protein_implicit_mindist,
                               ds_name=DS_NAME, single_procss=False)
runner.run_chunks()
runner.run_collate()
