from tqdm import tqdm

from geometry_processors.lm.esm_embedding import ESMCalculator
from geometry_processors.pl_dataset.prot_utils import pdb2seq
from geometry_processors.pl_dataset.yaowen_nonbinder_reader import NonbinderReader

reader = NonbinderReader()
seq_info = {}
for uniprot_id in tqdm(reader.uniprot_ids, desc="protein-seq"):
    try:
        seq = pdb2seq(reader.uniprot_id2prot_polarh(uniprot_id))
    except OSError as e:
        print(e)
        continue
    seq_info[uniprot_id] = seq

calc = ESMCalculator("/vast/sx801/geometries/Yaowen_nonbinders/prot_polar.esm2_t33_650M_UR50D")
calc.run(seq_info)