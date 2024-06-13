from typing import Union, Optional
from prody import AtomGroup, parsePDB, parsePDBStream
import torch as th
import numpy as np

INF_DIST = 1000000.

def pdb2seq(pdb_f: Optional[str] = None, pdb_stream = None, prot_ag: AtomGroup = None):
    if pdb_f is not None:
        assert prot_ag is None
        prot_ag = parsePDB(pdb_f).protein.toAtomGroup()
    if pdb_stream is not None:
        assert prot_ag is None
        prot_ag = parsePDBStream(pdb_stream).protein.toAtomGroup()
    out = ""
    for res in prot_ag.iterResidues():
        out += res.getSequence()[0]
    return out

def pdb2chain_seqs(pdb_f: str):
    prot_ag = parsePDB(pdb_f).protein.toAtomGroup()
    seqs = []
    for chain in prot_ag.iterChains():
        seqs.append(pdb2seq(prot_ag=chain.toAtomGroup()))
    return seqs

def pl_min_dist_matrix(lig_pos, prot_pos, device: str = "cpu"):
    """
    Calculate ligand-atom to protein residue minimum distance matrix.

    Input should be either numpy arrays or torch tensors;

    lig_pos should be [N_lig, 3] or [batch_size, N_lig, 3];
    prot_pos should be [N_aa, N_max_atom_per_aa, 3] or 
        [batch_size, N_aa, N_max_atom_per_aa, 3], respectively.
    """
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    if isinstance(lig_pos, np.ndarray):
        lig_pos: th.Tensor = th.as_tensor(lig_pos)
    if isinstance(prot_pos, np.ndarray):
        prot_pos: th.Tensor = th.as_tensor(prot_pos)
    lig_pos = lig_pos.double().to(device)
    prot_pos = prot_pos.double().to(device)

    N_l = lig_pos.shape[0]
    N_max = prot_pos.shape[-2]

    assert lig_pos.dim() == prot_pos.dim() - 1, f"error sizes: {lig_pos.shape}, {prot_pos.shape}"
    if lig_pos.dim() == 2:
        lig_pos = lig_pos.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = lig_pos.shape[0]
    prot_pos = prot_pos.view(batch_size, -1, 3)
    
    dists = -2 * th.bmm(lig_pos, prot_pos.permute(0, 2, 1)) + th.sum(prot_pos**2,    axis=-1).unsqueeze(1) + th.sum(lig_pos**2, axis=-1).unsqueeze(-1)	
    return th.nan_to_num((dists**0.5).view(batch_size, N_l,-1,N_max),INF_DIST).min(axis=-1)[0]

def pp_min_dist_matrix_vec(prot_pos):
    """
    Calculate protein residue-residue minimum distance matrix.
    prot_pos should be [N_aa, N_max_atom_per_aa, 3] or [batch_size, N_aa, N_max_atom_per_aa, 3].

    return: [batch_size, N_aa, N_aa]
    """
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    if isinstance(prot_pos, np.ndarray):
        prot_pos = th.as_tensor(prot_pos)
    prot_pos = prot_pos.double()

    if prot_pos.dim() == 3:
        prot_pos = prot_pos.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = prot_pos.shape[0]

    N_max = prot_pos.shape[-2]
    N_aa = prot_pos.shape[-3]
    prot_pos = prot_pos.view(batch_size, -1, 3)
    dists = -2 * th.bmm(prot_pos, prot_pos.permute(0, 2, 1)) + th.sum(prot_pos**2,    axis=-1).unsqueeze(1) + th.sum(prot_pos**2, axis=-1).unsqueeze(-1)	
    dists = th.nan_to_num((dists**0.5).view(batch_size, N_aa, N_max, N_aa, N_max), INF_DIST)
    min_dist = dists.min(axis=-1)[0].min(axis=-2)[0]
    return min_dist

def pp_min_dist_matrix_vec_mem(prot1_pos: Union[np.ndarray, th.Tensor], prot2_pos: Union[np.ndarray, th.Tensor] = None) -> th.Tensor:
    """
    WINNER!!!
    
    A memory efficient (but slower) implementation of PP min dist calculation.

    WARNING: only works for proteins
    """
    if isinstance(prot1_pos, np.ndarray):
        prot1_pos = th.as_tensor(prot1_pos)
    if isinstance(prot2_pos, np.ndarray):
        prot2_pos = th.as_tensor(prot2_pos)
    # self-self interaction
    if prot2_pos is None:
        prot2_pos = prot1_pos

    # the input shape should be: [N_aa, N_max, 3]
    assert prot1_pos.dim() == 3, prot1_pos.shape
    assert prot2_pos.dim() == 3, prot2_pos.shape

    n_aas = prot1_pos.shape[0]
    pp_min_martix = []
    for aa_id in range(n_aas):
        selec_aa = prot1_pos[aa_id, :, :]
        selec_pp = pl_min_dist_matrix(selec_aa, prot2_pos)
        this_pp_min = selec_pp.min(dim=1)[0]
        pp_min_martix.append(this_pp_min)
    # output dimension should be [N_aa_prot1, N_aa_prot2]
    out = th.concat(pp_min_martix, dim=0)
    return out

def pp_min_dist_naive(atomgroup):
    from prody import buildDistMatrix
    residues = [res for res in atomgroup.iterResidues()]
    min_dist = []
    for res1 in residues:
        this_min_dist = []
        for res2 in residues:
            this_matrix = buildDistMatrix(res1, res2)
            this_min_dist.append(np.min(this_matrix))
        min_dist.append(this_min_dist)
    return np.asarray(min_dist)

def pp_min_dist_oneway(atomgroup):
    from prody import buildDistMatrix
    residues = [res for res in atomgroup.iterResidues()]
    n_res = len(residues)
    min_dist = []
    for res1_id in range(n_res):
        this_min_dist = []
        for res2_id in range(n_res):
            # since it is a symetric matrix, we only calculate half of them
            # since we want to remove self-interaction, I set the diagnol elements to INF as well
            if res2_id <= res1_id:
                this_min_dist.append(INF_DIST)
                continue
            this_matrix = buildDistMatrix(residues[res1_id], residues[res2_id])
            this_min_dist.append(np.min(this_matrix))
        min_dist.append(this_min_dist)
    return np.asarray(min_dist)

def test_pp_min_dist():
    import time
    from geometry_processors.pl_dataset.ConfReader import PDBReader
    pdb_reader = PDBReader("/vast/sx801/geometries/PDBBind2020_OG/RenumPDBs/10gs.renum.pdb")
    protein_pad_dict = pdb_reader.get_padding_style_dict()

    # tik = time.time()
    # min_dist_naive = pp_min_dist_naive(pdb_reader.prody_parser)
    # tok = time.time()
    # t_naive = tok - tik
    # min_dist_naive = th.as_tensor(min_dist_naive)

    tik = time.time()
    min_dist_naive = pp_min_dist_matrix_vec_mem(protein_pad_dict["R"])
    tok = time.time()
    t_naive = tok - tik
    min_dist_naive = th.as_tensor(min_dist_naive)

    tik = time.time()
    min_dist = pp_min_dist_matrix_vec(protein_pad_dict["R"])
    tok = time.time()
    t_vec = tok - tik
    min_dist = min_dist.squeeze(0)

    diff = min_dist - min_dist_naive
    print(f"Diff: {diff}")
    print(f"Error: {diff.abs().sum()}")
    print(f"mem time: {t_naive}")
    print(f"Vec time: {t_vec}")

if __name__ == "__main__":
    seqs = pdb2chain_seqs("/scratch/sx801/temp/2ymd_protein.polar.pdb")
    print(len("".join(seqs)))
    seq_long = pdb2seq("/scratch/sx801/temp/2ymd_protein.polar.pdb")
    print(len(seq_long))
