from tqdm import tqdm

from geometry_processors.lm.esm_embedding import ESMCalculator
from geometry_processors.pl_dataset.prot_utils import pdb2seq
from geometry_processors.pl_dataset.yaowen_nonbinder_reader import NonbinderReader

reader = NonbinderReader()
seq_info = {}
pdb_ids = set(reader.sampled_pl_info_df["pdb_id"].values.reshape(-1).tolist())
for pdb_id in tqdm(pdb_ids, desc="protein-seq"):
    try:
        seq = pdb2seq(reader.pdb2prot_poarlh(pdb_id))
    except OSError as e:
        print(e)
        continue
    seq_info[pdb_id] = seq

calc = ESMCalculator("/vast/sx801/geometries/Yaowen_nonbinders/prot_crys_polar.esm2_t33_650M_UR50D")
calc.run(seq_info)