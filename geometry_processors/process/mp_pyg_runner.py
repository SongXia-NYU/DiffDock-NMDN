import logging
from typing import List, Callable
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from geometry_processors.misc import solv_num_workers
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.all2single_pygs import PLHeteroProcessor, PLMinDistImplicitProcessor, PLProcessor
import torch
from glob import glob
from torch_geometric.data import Data, InMemoryDataset
from tqdm.contrib.concurrent import process_map

from geometry_processors.process.concat_pyg import concat_pyg
from geometry_processors.process.pp_processors import ProtProtIntraProcessor, ProtProtProcessor
from geometry_processors.process.single_prot_processor import SingleProtProcessor

CHUNK_ROOT = "/vast/sx801/single_pygs"
SAVE_ROOT = "/scratch/sx801/data/im_datasets/processed"

class DataModifier:
    def __call__(self, data: Data) -> Data:
        raise NotImplementedError

class PygRunnerBase:
    def __init__(self, info_list: List[MPInfo], proc_fn: Callable[[MPInfo], Data], 
                 ds_name: str, chunk_ckpt: bool=False, 
                 chunk_root=None, save_root=None, data_modifier: DataModifier = None) -> None:
        chunk_root = chunk_root if chunk_root else CHUNK_ROOT
        save_root = save_root if save_root else SAVE_ROOT

        self.info_list = info_list
        # see proc_implicit_mindist as an example
        self.proc_fn = proc_fn
        self.ds_name = ds_name
        self.chunk_ckpt = chunk_ckpt

        self.chunk_folder = osp.join(chunk_root, self.ds_name)
        os.makedirs(self.chunk_folder, exist_ok=True)
        self.save_pyg = osp.join(save_root, f"{self.ds_name}.pyg")
        # only used when saving multiple parts
        self.save_parts_root = osp.join(save_root, f"{self.ds_name}")
        self.data_modifier = data_modifier

    def run_collate(self, nparts: int=None, check: bool = False):
        chunks = glob(osp.join(self.chunk_folder, "chunk_*.pth"))

        if nparts is None:
            return self._collate_chunks(chunks, self.save_pyg, check)

        # when the dataset is too large to load in memory, use mutiple parts
        os.makedirs(self.save_parts_root, exist_ok=True)
        chunk_o_chunks = np.array_split(chunks, nparts)
        for part_id, this_chunks in enumerate(chunk_o_chunks):
            self._collate_chunks(this_chunks, osp.join(self.save_parts_root, f"part_{part_id}.pyg"), check)

    def _collate_chunks(self, chunks: List[str], save_pyg: str, check: bool):
        data_list = []
        for chunk in chunks:
            this_dl = torch.load(chunk)
            # for d in this_dl[1:]:
            #     if d is None: continue
            #     del d[("protein", "interaction", "protein")]
            data_list.extend(this_dl)
        if check:
            data_list = [d for d in data_list if self.check_sanity(d)]
        data_list = [d for d in data_list if d]
        if self.data_modifier is not None:
            data_list = [self.data_modifier(d) for d in data_list]
        data_list = [d for d in data_list if d]
        n_max_pad = self.determin_padding()
        concat_pyg(data_list=data_list, save_pyg=save_pyg, n_max_pad=n_max_pad)

    def determin_padding(self):
        if not self.info_list:
            return 44
        n_max_pad = 44 if self.info_list[0].protein_pdb is not None else None
        return n_max_pad

    def check_sanity(self, d: Data):
        n_lig = d.N
        n_aa = d.N_prot.shape[0]
        pl_edge = d.PL_min_dist_sep_oneway_edge_index
        lig_idx = pl_edge[0]
        prot_idx = pl_edge[1]

        success = True
        if n_lig != d.R.shape[0]:
            logging.warning(f"Inconsistent N: {n_lig} vs. {d.R.shape}")
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


class MultiProcessPygRunner(PygRunnerBase):
    def __init__(self, desc_prefix: str = None, single_procss: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.desc_prefix = desc_prefix
        self.single_procss: bool = single_procss

    def run_debug(self):
        for info in self.info_list[:5]:
            d = self.proc_fn(info)
            breakpoint()

    def run(self):
        self.run_chunks()
        self.run_collate()

    def run_chunks(self):
        os.makedirs(self.chunk_folder, exist_ok=True)
        CHUNK_SIZE = 3000

        info_chunks = []
        tmp_info_list = []

        for this_info in self.info_list:
            tmp_info_list.append(this_info)
            if len(tmp_info_list) == CHUNK_SIZE:
                info_chunks.append(tmp_info_list)
                tmp_info_list = []
        info_chunks.append(tmp_info_list)
        __, __, n_workers = solv_num_workers()
        for i, chunk in enumerate(info_chunks):
            chunk_path = osp.join(self.chunk_folder, f"chunk_{i}.pth")
            if self.chunk_ckpt and osp.exists(chunk_path):
                continue
            desc_prefix = "" if self.desc_prefix is None else self.desc_prefix
            desc = desc_prefix + f"Chunk {i+1}/{len(info_chunks)}"
            if self.single_procss:
                data_list = [self.proc_fn(c) for c in tqdm(chunk, desc=desc)]
            else:
                # multi-process by tqdm process_map
                data_list = process_map(self.proc_fn, chunk, max_workers=n_workers, desc=desc, chunksize=5)

            torch.save(data_list, chunk_path)
            del data_list


class ArrayPygRunner(PygRunnerBase):
    """
    In the extreme cases when multi-processing cannot satisfy you, array jobs are needed.
    """
    def __init__(self, array_id: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.array_id = array_id
    
    def run_array(self, collate: bool = False):
        __, __, n_workers = solv_num_workers()
        # assert n_workers == 1, "Use single processor for array jobs"

        chunk_path = osp.join(self.chunk_folder, f"chunk_{self.array_id}.pth")
        if osp.exists(chunk_path):
            print(f"{chunk_path} exists. exiting...")
            return

        data_list = []
        for info in tqdm(self.info_list, desc=f"array {self.array_id}"):
            data_list.append(self.proc_fn(info))
        if collate:
            data_list = InMemoryDataset.collate(data_list)
        torch.save(data_list, chunk_path)


def proc_implicit_mindist(info: MPInfo) -> Data:
    """
    Protein: implicit
    Ligand: explicit
    edges: PL_min_dist_edge
    """
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    processor = PLMinDistImplicitProcessor(info, protein_reader_args={"dry": True}, cal_pp=False)
    try:
        d = processor.process_single_entry()
        return d
    except Exception as e:
        print(f"Error processing {vars(info)}:", e)
        return None

def proc_hetero_graph(info: MPInfo) -> Data:
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    processor = PLHeteroProcessor(info, protein_reader_args={"force_polarh": True})
    try:
        d = processor.process_single_entry()
        return d
    except Exception as e:
        raise e
        print(f"Error processing {vars(info)}:", e)
        return None

def proc_ligand(info: MPInfo) -> Data:
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    assert info.protein_pdb is None, info
    processor = PLProcessor(info)
    try:
        d = processor.process_single_entry(save=False)
        return d
    except Exception as e:
        print(f"Error processing {vars(info)}:", e)
        return None
    
def proc_pp_implicit_mindist(info: MPInfo) -> Data:
    """
    Protein: implicit
    Ligand: explicit
    edges: PL_min_dist_edge
    """
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    processor = ProtProtProcessor(info, protein_reader_args={"dry": True})
    try:
        d = processor.process_single_entry()
        return d
    except Exception as e:
        print(f"Error processing {vars(info)}:", e)
        raise e
        return None

def proc_pp_intra_implicit_mindist(info: MPInfo) -> Data:
    """
    Protein: implicit
    Ligand: explicit
    edges: PL_min_dist_edge
    """
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    processor = ProtProtIntraProcessor(info, protein_reader_args={"dry": True})
    try:
        d = processor.process_single_entry()
        return d
    except Exception as e:
        print(f"Error processing {vars(info)}:", e)
        raise e
    
def proc_single_protein_implicit_mindist(info: MPInfo) -> Data:
    """
    Protein: implicit
    Ligand: explicit
    edges: PL_min_dist_edge
    """
    # disable the logging from prody
    logger = logging.getLogger(".prody")
    logger.setLevel(logging.CRITICAL)

    processor = SingleProtProcessor(info, protein_reader_args={"dry": True, "force_polarh": False})
    try:
        d = processor.process_single_entry()
        return d
    except Exception as e:
        print(f"Error processing {vars(info)}:", e)
        # raise e
