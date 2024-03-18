import logging
import os
import os.path as osp
from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from glob import glob

from utils.data.DummyIMDataset import AuxPropDataset, DummyIMDataset
from utils.data.data_utils import get_lig_coords, infer_device, infer_type, get_lig_natom, get_pl_edge, get_prot_coords, get_prot_natom
from utils.utils_functions import get_device, floating_type


class ProteinEmbeddingDS(AuxPropDataset):
    """
    Inject protein embedding (calculated by large language models such as ESM) into the training set.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prot_embedding_root = self.cfg["prot_embedding_root"]
        if not osp.exists(self.prot_embedding_root):
            raise ValueError("Invalid prot_embedding_root: "+self.prot_embedding_root)
        self.prot_embed_use_chunks = self.cfg["prot_embed_use_chunks"]
            
        self._sanity_checked_ids = set()

        self.load_prot_embed()

    def load_prot_embed(self):
        # chunk behaviour: multiple protein one file
        if not self.prot_embed_use_chunks:
            return self.load_prot_embed_legacy()
        
        # only loaded needed pdbs' ESM embedding to save memory
        needed_pdbs = set()
        for split_name in ["train_index", "val_index", "test_index"]:
            this_index = getattr(self, split_name)
            if this_index is None:
                continue
            for idx in this_index:
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                needed_pdbs.add(self.idx2pdb_mapper[idx])

        # load ESM embedding from memory, again, only needed protein embeddings are loaded
        prot_embedding_mapper = {}
        chunks = glob(osp.join(self.prot_embedding_root, "chunk_*.pth"))
        is_deep_acc_net: bool = "/DeepAccNet/" in self.prot_embedding_root
        for chunk in tqdm(chunks, desc="loading ESM chunks"):
            chunk_dict: Dict[str, torch.Tensor] = torch.load(chunk, map_location="cpu")
            chunk_dict = {key.split(".")[0]: chunk_dict[key] for key in chunk_dict}
            if is_deep_acc_net:
                # eg: "3gvaA_protein" -> "3gvaA"
                chunk_dict = {key.split("_")[0]: chunk_dict[key] for key in chunk_dict}
            this_needed_pdbs = set([key.split(".")[0] for key in chunk_dict]).intersection(needed_pdbs)
            loaded_chunk_dict = {pdb: chunk_dict[pdb].cpu().squeeze(0)[1: -1, :].type(floating_type) 
                                 for pdb in this_needed_pdbs}
            prot_embedding_mapper.update(loaded_chunk_dict)
            # release memory
            del chunk_dict
        self.prot_embedding_mapper = prot_embedding_mapper

        self.cleanup_indices()
        return
    
    def cleanup_indices(self):
        # cleanup indices that do not have protein embedding
        for split_name in ["train_index", "val_index", "test_index"]:
            this_index = getattr(self, split_name)
            if this_index is None:
                continue
            index_clean = []
            for i in tqdm(this_index, desc="clean_embed_file"):
                i = i.item()
                pdb = self.idx2pdb_mapper[i]

                if pdb not in self.prot_embedding_mapper:
                    continue
                index_clean.append(i)

            index_clean = torch.as_tensor(index_clean)

            n_before = len(this_index)
            n_after = len(index_clean)
            assert n_after > int(0.9 * n_before), \
                f"More than 10% indices removed due to protein embedding, before: {n_before}, after: {n_after}"
            setattr(self, split_name, index_clean)
        return
    
    def load_prot_embed_legacy(self):
        # remove the indices that do not have ESM embeddings due to OOM errors
        # and load needed embeddings into memory
        prot_embedding_mapper = {}
        for split_name in ["train_index", "val_index", "test_index"]:
            this_index = getattr(self, split_name)
            if this_index is None:
                continue
            index_clean = []
            for i in tqdm(this_index, desc="clean_embed_file"):
                i = i.item()
                pdb = self.idx2pdb_mapper[i]
                # avoid duplicate interaction with the file system
                if pdb in prot_embedding_mapper:
                    index_clean.append(i)
                    continue

                prot_embed = self.try_load_prot_embed(pdb)
                if prot_embed is None:
                    continue
                index_clean.append(i)
                # remove the start-of-sentence token and end-of-sentance token
                prot_embedding_mapper[pdb] = prot_embed.type(floating_type)

            index_clean = torch.as_tensor(index_clean)

            n_before = len(this_index)
            n_after = len(index_clean)
            assert n_after > int(0.9 * n_before), \
                f"More than 10% indices removed due to protein embedding, before: {n_before}, after: {n_after}"

            setattr(self, split_name, index_clean)
        self.prot_embedding_mapper = prot_embedding_mapper
        return

    def try_load_prot_embed(self, pdb):
        # chunk behaviour: multiple protein one file
        if self.prot_embed_use_chunks:
            if pdb not in self.pdb2chunk:
                return None
            
            tgt_chunk_id = self.pdb2chunk[pdb]
            if tgt_chunk_id == self.loaded_chunk_id:
                if pdb not in self.loaded_chunk_dict:
                    return None
                return self.loaded_chunk_dict[pdb].cpu().squeeze(0)[1: -1, :]
            
            chunk_dict = torch.load(osp.join(self.prot_embedding_root, f"chunk_{tgt_chunk_id}.pth"))
            self.loaded_chunk_dict = {key.split(".")[0]: chunk_dict[key] for key in chunk_dict}
            self.loaded_chunk_id = tgt_chunk_id
            if pdb not in self.loaded_chunk_dict:
                return None
            return self.loaded_chunk_dict[pdb].cpu().squeeze(0)[1: -1, :]
        
        # legacy behaviour: one protein one file
        # due to different naming convention, I need to check different combination of files :<
        file_names = [f"{pdb}.pth", f"{pdb}_protein.pth"]
        embed_roots = self.prot_embedding_root.split(";")
        for embed_root in embed_roots:
            for file_name in file_names:
                try_file = osp.join(embed_root, file_name)
                if osp.exists(try_file):
                    # remove the start-of-sentanse token and end-of-sentance token
                    return torch.load(try_file, map_location="cpu").squeeze(0)[1: -1, :]
        return None

    def get(self, idx: int, process=True) -> Data:
        #TODO: check 6gga: prot_embed-196
        d = super().get(idx, process)
        pdb = self.idx2pdb_mapper[idx]
        d.prot_embed = self.prot_embedding_mapper[pdb].to(infer_device(d)).type(floating_type)
        d = self.proc4water(d)

        # check the sanity of the data. Only check once
        if idx in self._sanity_checked_ids:
            return d
        if not self.check_sanity(d):
            raise ValueError(f"Failed sanity check at idx {idx}. Check training.log for more details.")
        self._sanity_checked_ids.add(idx)
        
        return d
    
    def check_sanity(self, d: Data):
        # single protein is currently not checked.
        if hasattr(d, "single_prot_identifier"):
            return True
        
        n_lig = get_lig_natom(d)
        n_aa = get_prot_natom(d).shape[0]
        pl_edge = get_pl_edge(d)
        lig_idx = pl_edge[0]
        prot_idx = pl_edge[1]

        success = True
        lig_coords = get_lig_coords(d)
        if n_lig != lig_coords.shape[0]:
            logging.warning(f"Inconsistent N: {n_lig} vs. {lig_coords.shape}")
            success = False
        if n_aa != d.prot_embed.shape[0]:
            logging.warning(f"Inconsistent N_prot: {n_aa} vs. {d.prot_embed.shape}")
            success = False
        if lig_idx.numel() > 0 and lig_idx.max() >= n_lig:
            logging.warning(f"Error ligand idx: {lig_idx.max()} >= {n_lig}")
            logging.warning(f"lig_idx: {lig_idx}")
            success = False
        if prot_idx.numel() > 0 and prot_idx.max() >= n_aa:
            logging.warning(f"Error protein idx: {prot_idx.max()} >= {n_aa}")
            logging.warning(f"prot_idx: {prot_idx}")
            success = False
        
        if not success:
            logging.warning("Failed sanity check :(")
            logging.warning(f"PDB: {d.pdb}")
            logging.warning(f"ligand_file: {d.ligand_file}")
            logging.warning(f"protein_file: {d.protein_file}")
        return success

    def proc4water(self, d: Data):
        # extra processing to remove water molecules in the pl_index
        n_aas = d.prot_embed.shape[0]
        n_res_from_r = get_prot_coords(d).shape[0]

        if n_aas == n_res_from_r:
            return d
        if n_res_from_r == 0:
            # this happens when I want to skip the protein information
            return d
        # Here I somehow counted water as residues which is not in the prot_embed, so I have to remove them
        assert n_aas < n_res_from_r, str(d)
        z_water = d.Z_prot_pad[n_aas:, :]
        # if the "residue" has only one atom, assumed to be metal ions, which should be removed before checking
        z_water = z_water[(z_water!=-1).sum(dim=-1)!=1, :]
        # make sure they are all water
        assert (z_water[:, 0] == 8).sum() == z_water.shape[0], z_water
        assert (z_water[:, 1] == 1).sum() == z_water.shape[0], z_water
        assert (z_water[:, 2] == 1).sum() == z_water.shape[0], z_water
        # water should be O (8), H (1) and H(1). The rest should be padding
        assert (z_water[:, 3] == -1).sum() == z_water.shape[0], z_water
        pl_index = d.PL_min_dist_sep_oneway_edge_index
        # remove the extra pl_index since we are removing protein atoms
        # pl_index is organised as [[lig_idx, prot_idx]]
        pl_index_mask = (pl_index[1] < n_aas)
        d.PL_min_dist_sep_oneway_edge_index = d.PL_min_dist_sep_oneway_edge_index[:, pl_index_mask]
        d.PL_min_dist_sep_oneway_dist = d.PL_min_dist_sep_oneway_dist[pl_index_mask]
        return d
    

class PPEmbedDS(ProteinEmbeddingDS):
    """
    Embedding data set for protein-protein interaction.
    """
    def __init__(self, **kwargs):
        self._embed_key_style2 = False
        super().__init__(**kwargs)
        # see self.get_chain_keys()

    def load_prot_embed(self):
        # chunk behaviour: multiple protein one file
        assert self.prot_embed_use_chunks
        
        # only loaded needed pdbs' ESM embedding to save memory
        needed_pdbs = set()
        for split_name in ["train_index", "val_index", "test_index"]:
            this_index = getattr(self, split_name)
            if this_index is None:
                continue
            for idx in this_index:
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                needed_pdbs.add(self.idx2pdb_mapper[idx])

        # load ESM embedding from memory, again, only needed protein embeddings are loaded
        prot_embedding_mapper = {}
        chunks = glob(osp.join(self.prot_embedding_root, "chunk_*.pth"))
        for chunk in tqdm(chunks, desc="loading ESM chunks"):
            # the keys are like: ['2j3k_chain_1', '4kwi_chain_1', '7dye_chain_1', '5c2u_chain_1', '7u58_chain_1', '6kvs_chain_2', ...]
            chunk_dict: Dict[str, torch.Tensor] = torch.load(chunk, map_location="cpu")
            self._embed_key_style2 = list(chunk_dict.keys())[0].split("_")[-1].startswith("chain")
            for pdb in needed_pdbs:
                chain1_key, chain2_key = self.get_chain_keys(pdb)
                if chain1_key in chunk_dict: prot_embedding_mapper[chain1_key] = chunk_dict[chain1_key].cpu().squeeze(0)[1: -1, :]
                if chain2_key in chunk_dict: prot_embedding_mapper[chain2_key] = chunk_dict[chain2_key].cpu().squeeze(0)[1: -1, :]
                
        # set the "pdb key" for the protein protein pairs with a dummy tensor.
        for pdb in needed_pdbs:
            chain1_key, chain2_key = self.get_chain_keys(pdb)
            if chain1_key in prot_embedding_mapper and chain2_key in prot_embedding_mapper:
                prot_embedding_mapper[pdb] = torch.zeros(1)
        self.prot_embedding_mapper = prot_embedding_mapper

        self.cleanup_indices()
        return
    
    def get(self, idx: int, process=True) -> Data:
        d: Data = super().get(idx, process)
        pdb = self.idx2pdb_mapper[idx]
        chain1_key, chain2_key = self.get_chain_keys(pdb)
        d.prot_embed_chain1 = self.prot_embedding_mapper[chain1_key].to(d.R.device).type(floating_type)
        d.prot_embed_chain2 = self.prot_embedding_mapper[chain2_key].to(d.R.device).type(floating_type)

        # check if the data is processed correctly
        if idx not in self._sanity_checked_ids:
            assert d.PP_min_dist_oneway_edge_index[0].max() < d.prot_embed_chain1.shape[0], str(d)
            assert d.PP_min_dist_oneway_edge_index[1].max() < d.prot_embed_chain2.shape[0], str(d)
            self._sanity_checked_ids.add(idx)

        d.atom_mol_batch_chain1 = torch.zeros(d.N_aa_chain1, device=d.R.device).long()
        d.atom_mol_batch = d.atom_mol_batch_chain1
        d.atom_mol_batch_chain2 = torch.zeros(d.N_aa_chain2, device=d.R.device).long()
        return d

    def get_chain_keys(self, pdb: str) -> Tuple[str, str]:
        # For some reason we have two ways to name the chains: chain_i and chaini
        # I wrote a simple detection system to automatically detect them.
        chain1_key: str = f"{pdb}_chain_1"
        chain2_key: str = f"{pdb}_chain_2"
        if self._embed_key_style2:
            chain1_key: str = f"{pdb}_chain1"
            chain2_key: str = f"{pdb}_chain2"
        return chain1_key, chain2_key
    
    def proc4water(self, d: Data):
        return d
    
    def check_sanity(self, d: Data):
        # sanity will be checked after infusing the protein embedding in self.get() function.
        return True
