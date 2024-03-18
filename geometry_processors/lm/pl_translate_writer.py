import logging
import os
import os.path as osp
import traceback

from prody import parsePDB
from rdkit.Chem import MolFromPDBFile, MolToSmiles

from geometry_processors.pl_dataset.ConfReader import PDBReader

class PLTranslateWriter:
    def __init__(self, save_root, chunksize=10_000, capped=False) -> None:
        # disable the logging from prody
        logger = logging.getLogger(".prody")
        logger.setLevel(logging.CRITICAL)
        
        self.save_root = save_root
        self.chunksize = chunksize
        self.capped = capped

        self._p2l_src_writer = None
        self._p2l_tgt_writer = None
        self._l2p_src_writer = None
        self._l2p_tgt_writer = None

        self._p2l_src_cache = ""
        self._p2l_tgt_cache = ""
        self._l2p_src_cache = ""
        self._l2p_tgt_cache = ""
        self._cache_counter = 0

    def proc_pdb(self, pdb_f):
        reader = PDBReader(pdb_f, capped=self.capped)
        seq = reader.sequence

        try:
            mol = MolFromPDBFile(pdb_f)
            smiles = MolToSmiles(mol)
        except Exception as e:
            print(e)
            traceback.print_exc()
            print(f"Error in entry {pdb_f}, returning...")
            return
        smiles = f"<mod>{smiles}</mod>"
        
        self._p2l_src_cache += f"SeqToSMILES:{seq}\n"
        self._p2l_tgt_cache += f"{smiles}\n"
        self._l2p_src_cache += f"SMILESToSeq:{smiles}\n"
        self._l2p_tgt_cache += f"{seq}\n"
        self._cache_counter += 1

        if self._cache_counter >= self.chunksize:
            self.write_cache()

    def write_cache(self):
        self.p2l_src_writer.write(self._p2l_src_cache)
        self.p2l_tgt_writer.write(self._p2l_tgt_cache)
        self.l2p_src_writer.write(self._l2p_src_cache)
        self.l2p_tgt_writer.write(self._l2p_tgt_cache)

        self._cache_counter = 0
        self._p2l_src_cache = ""
        self._p2l_tgt_cache = ""
        self._l2p_src_cache = ""
        self._l2p_tgt_cache = ""

    @property
    def p2l_src_writer(self):
        if self._p2l_src_writer is None:
            self._p2l_src_writer = self.init_a_writer("p2l_src.txt")
        return self._p2l_src_writer

    @property
    def p2l_tgt_writer(self):
        if self._p2l_tgt_writer is None:
            self._p2l_tgt_writer = self.init_a_writer("p2l_tgt.txt")
        return self._p2l_tgt_writer

    @property
    def l2p_src_writer(self):
        if self._l2p_src_writer is None:
            self._l2p_src_writer = self.init_a_writer("l2p_src.txt")
        return self._l2p_src_writer

    @property
    def l2p_tgt_writer(self):
        if self._l2p_tgt_writer is None:
            self._l2p_tgt_writer = self.init_a_writer("l2p_tgt.txt")
        return self._l2p_tgt_writer

    def init_a_writer(self, name):
        os.makedirs(self.save_root, exist_ok=True)
        p = osp.join(self.save_root, name)
        assert not osp.exists(p), f"{p} exists..."
        return open(p, "w")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.write_cache()
        self.p2l_src_writer.close()
        self.p2l_tgt_writer.close()
        self.l2p_src_writer.close()
        self.l2p_tgt_writer.close()

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            # return False # uncomment to pass exception through
        
        return True
