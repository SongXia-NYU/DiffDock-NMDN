import argparse
import torchvision

import rdkit
from rdkit.Chem import MolFromPDBFile
from Networks.esm_gearnet.pl_dataset import ESMGearNetProtLig

# ['1gpk', '3gr2', '4k18']
# ['1gpk', '1h22', '1h23', '1oyt', '1p1n', '1p1q', '1r5y', 
# '2c3i', '2j7h', '2qe4', '2v7a', '2wer', '2wn9', '2wnc', 
# '2x00', '2xb8', '2xbv', '2xdl', '2y5h', '2ymd', '2zda', 
# '3arq', '3arv', '3ary', '3bv9', '3g0w', '3gc5', '3ge7', 
# '3gr2', '3n76', '3n7a', '3n86', '3o9i', '3rr4', '3ryj', 
# '3u5j', '3u8k', '3u8n', '3utu', '4bkt', '4ciw', '4k18', 
# '4mgd', '4qac', '4ty7', '4w9c', '4w9h', '4w9l']
# /CASF-2016-cyang/coreset_noh/1gpk/1gpk_protein.noh.pdb
from geometry_processors.pl_dataset.casf2016_reader import CASF2016Reader
reader = CASF2016Reader("/CASF-2016-cyang")
for pdb in ['1gpk', '3gr2', '4k18']:
    noh_pdb = reader.pdb2prot_noh(pdb)
    noh_pdb = f"/scratch/sx801/temp/{pdb}.pdbfixer.pdb"
    # print(noh_pdb)
    mol = MolFromPDBFile(noh_pdb)
    print(pdb, mol)
# exit()

parser = argparse.ArgumentParser()
parser.add_argument("--array_id", type=int)
args = parser.parse_args()
array_id: int = args.array_id

ds_args = {"data_root": "/scratch/sx801/data/im_datasets/",
           "dataset_name": "casf-scoring.prot.polar.lig.polar.implicit.min_dist.pyg",
           "split": None,
           "test_name": "hetero.polar.polar.implicit.min_dist",
           "config_args": None}
ds = ESMGearNetProtLig(array_id, **ds_args)
breakpoint()
