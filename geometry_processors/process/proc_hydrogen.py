import subprocess
from tempfile import TemporaryDirectory
import os.path as osp
from typing import List
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import json
import signal

from geometry_processors.misc import solv_num_workers

if osp.exists("/scratch/sx801/"):
    SCRIPT_ROOT = "/softwares"
    DS_ROOT = "/vast/sx801"
else:
    SCRIPT_ROOT = "/home/carrot_of_rivia/Documents/PycharmProjects"
    DS_ROOT = "/home/carrot_of_rivia/Documents/disk/datasets"
PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor"
PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_ligand"

def handler(signum, frame):
    raise RuntimeError("TIMEOUT running PREPARE_LIG")

class BaseConverter:
    def __init__(self, f_in: str, f_out: str) -> None:
        self.f_in = f_in
        self.f_out = f_out

    def run(self):
        try:
            self._run()
            return None
        except Exception as e:
            print(e)
            signal.alarm(0)
            return {"f_in": self.f_in, "f_out": self.f_out, "error_msg": str(e)}
        
    def _run(self):
        raise NotImplementedError


class LigPolarConverter(BaseConverter):
    """
    Remove the non-polar Hydrogens of a ligand molecule.
    Input can be pdb mol2 or sdf format, output should be sdf/mol2 format.
    """
    def __init__(self, f_in: str, f_out: str, f_pdbqt: str = None) -> None:
        super().__init__(f_in, f_out)
        self.ext_in = f_in.split(".")[-1]
        assert self.ext_in in ["sdf", "mol2", "pdb"], self.f_in

        self.file_handle = osp.basename(f_in).split(f".{self.ext_in}")[0]

        self.ext_out = f_out.split(".")[-1]
        assert self.ext_out in ["sdf", "mol2"], self.f_out
        
        self.f_pdbqt = f_pdbqt

    
    def _run(self):
        if osp.exists(self.f_out): return
        with TemporaryDirectory() as d:
            # The input molecule must be in MOL2 format
            if self.ext_in != "mol2":
                mol2_file = osp.join(d, f"{self.file_handle}.mol2")
                subprocess.run(f"obabel -i{self.ext_in} {self.f_in} -omol2 -O {mol2_file}", shell=True, check=True)
            else:
                mol2_file = self.f_in
            # MOL2 -> PDBQT
            pdbqt_file = osp.join(d, f"{self.file_handle}.pdbqt") if self.f_pdbqt is None else self.f_pdbqt
            if not osp.exists(pdbqt_file):
                workdir = osp.dirname(mol2_file)
                # Due to some weird bug, PREPARE_LIG hangs when processing some molecules: e.g. C-C#N
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(40)
                subprocess.run(f"{PREPARE_LIG} -l {mol2_file} -U nphs_lps -A 'checkhydrogens' -o {pdbqt_file} ", 
                               shell=True, check=True, cwd=workdir)
                signal.alarm(0)
            # Convert PDBQT file (only polarH) back to output format
            subprocess.run(f"obabel -ipdbqt {pdbqt_file} -o{self.ext_out} -O {self.f_out} --title {self.f_in}", 
                           shell=True, check=True)


class ProtPolarConverter(BaseConverter):
    """
    Remove the non-polar Hydrogens of proteins/peptides.
    input and output should be PDB format.
    """
    def __init__(self, f_in: str, f_out: str) -> None:
        f_in = osp.abspath(f_in)
        super().__init__(f_in, f_out)
        self.ext_in = f_in.split(".")[-1]
        assert self.ext_in in ["pdb"], self.f_in

        self.file_handle = osp.basename(f_in).split(f".{self.ext_in}")[0]

        self.ext_out = f_out.split(".")[-1]
        assert self.ext_out in ["pdb"], self.f_out

    def _run(self):
        with TemporaryDirectory() as d:
            pdbqt_file = osp.join(d, f"{self.file_handle}.pdbqt")
            subprocess.run(f"cd {d}; {PREPARE_PROT} -r {self.f_in} -U nphs_lps -A 'checkhydrogens' -o {pdbqt_file} ", shell=True, check=True)
            subprocess.run(f"obabel -ipdbqt {pdbqt_file} -o{self.ext_out} -O {self.f_out}", shell=True, check=True)


class HydrogenRemover(BaseConverter):
    """
    Remove hydrogen atom from protein or ligand. Works for common format like PDB, SDF or MOL2.
    """
    def __init__(self, f_in: str, f_out: str) -> None:
        super().__init__(f_in, f_out)

        self.fmt_in = f_in.split(".")[-1]
        self.fmt_out = f_out.split(".")[-1]

    def run(self):
        cmd = f"obabel -i{self.fmt_in} {self.f_in} -o{self.fmt_out} -O {self.f_out} -d"
        subprocess.run(cmd, shell=True, check=True)

class HydrogenAdder(BaseConverter):
    """
    Add hydrogen atom to protein or ligand. Works for common format like PDB, SDF or MOL2.
    """
    def __init__(self, f_in: str, f_out: str) -> None:
        super().__init__(f_in, f_out)

        self.fmt_in = f_in.split(".")[-1]
        self.fmt_out = f_out.split(".")[-1]

    def _run(self):
        cmd = f"obabel -i{self.fmt_in} {self.f_in} -o{self.fmt_out} -O {self.f_out} -h"
        subprocess.run(cmd, shell=True, check=True)

def runner(conv: BaseConverter):
    return conv.run()


class BatchRunner:
    def __init__(self, error_file: str, converters: List[BaseConverter]) -> None:
        self.error_file = error_file
        self.converters = converters

    def run(self):
        __, __, n_workers = solv_num_workers()

        prot_errors = process_map(runner, self.converters, chunksize=5, max_workers=n_workers+2)
        prot_errors = [e for e in prot_errors if e]
        with open(self.error_file, "w") as f:
            json.dump(prot_errors, f, indent=2)

    def run_single_cpu(self):
        prot_errors = [runner(conv) for conv in tqdm(self.converters)]
        prot_errors = [e for e in prot_errors if e]
        with open(self.error_file, "w") as f:
            json.dump(prot_errors, f, indent=2)
