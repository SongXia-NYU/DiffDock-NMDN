import os
import os.path as osp
import signal
import subprocess
from tempfile import TemporaryDirectory

from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.process.proc_hydrogen import PREPARE_PROT, PREPARE_LIG, LigPolarConverter, handler


LINF9_PATH = "/softwares/Lin_F9_test"
LINF9 = f"{LINF9_PATH}/smina.static"

class LinF9LocalOptimizer:
    def __init__(self, info: MPInfo) -> None:
        self.protein_pdb = info.protein_pdb
        self.ligand_mol2 = info.ligand_mol2
        self.ligand_sdf = info.ligand_sdf

        self.protein_pdbqt = info.protein_pdbqt
        self.ligand_pdbqt = info.ligand_pdbqt
        self.ligand_linf9_opt = info.ligand_linf9_opt

    def run(self):
        if not osp.exists(self.protein_pdbqt):
            assert self.protein_pdb is not None
            subprocess.run(f"{PREPARE_PROT} -r {self.protein_pdb} -U nphs_lps -A 'checkhydrogens' -o {self.protein_pdbqt} ", shell=True, check=True)
        if not osp.exists(self.ligand_pdbqt):
            ligand_mol2 = self.ligand_mol2
            if self.ligand_mol2 is None:
                assert self.ligand_sdf is not None
                temp_dir = TemporaryDirectory()
                mol2_file = osp.join(temp_dir.name, "tmp.mol2")
                conv_cmd = f"obabel -isdf {self.ligand_sdf} -omol2 -O {mol2_file}"
                print(conv_cmd)
                subprocess.run(conv_cmd, shell=True, check=True)
                ligand_mol2 = mol2_file
            ligand_dir = osp.dirname(ligand_mol2)
            lig_cmd = f"{PREPARE_LIG} -l {ligand_mol2} -U nphs_lps -A 'checkhydrogens' -o {self.ligand_pdbqt} "
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(10)
            subprocess.run(lig_cmd, shell=True, check=True, cwd=ligand_dir)
            signal.alarm(0)
            if "temp_dir" in locals():
                temp_dir.cleanup()
        linf9_cmd = f"{LINF9} -r {self.protein_pdbqt} -l {self.ligand_pdbqt} --local_only --scoring Lin_F9 -o {self.ligand_linf9_opt} "
        subprocess.run(linf9_cmd, shell=True, check=True)


class LinF9OptConverter:
    """
    Tempory pdbqt structures.
    """
    def __init__(self, src_prot_pdb: str, src_lig_mol2: str, dst_lig_pdb: str, lig_pdbqt: str = None) -> None:
        self.src_prot_pdb = src_prot_pdb
        self.src_lig_mol2 = src_lig_mol2
        self.dst_lig_pdb = dst_lig_pdb

        self.lig_pdbqt = lig_pdbqt

    def run(self):
        temp_dir = TemporaryDirectory()
        prot_pdbqt = osp.join(temp_dir.name, "temp.prot.pdbqt")
        lig_pdbqt = osp.join(temp_dir.name, "temp.lig.pdbqt") if self.lig_pdbqt is None else self.lig_pdbqt
        info = MPInfo(protein_pdb=self.src_prot_pdb, ligand_mol2=self.src_lig_mol2, protein_pdbqt=prot_pdbqt,
                      ligand_pdbqt=lig_pdbqt, ligand_linf9_opt=self.dst_lig_pdb)
        optimizer = LinF9LocalOptimizer(info)
        try:
            optimizer.run()
        except Exception as e:
            print(e)
            return {"f_in": self.src_lig_mol2, "f_out": self.dst_lig_pdb, "error_msg": str(e)}
        temp_dir.cleanup()
        return

