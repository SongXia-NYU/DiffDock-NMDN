import os
import os.path as osp
import subprocess
from typing import List
from glob import glob
from geometry_processors.pl_dataset.csv2input_list import MPInfo
from geometry_processors.pl_dataset.linf9_local_optimizer import LINF9
from geometry_processors.process.proc_hydrogen import PREPARE_PROT, PREPARE_LIG


class LinF9GlobalOptimizer:
    def __init__(self, prot_pdb: str, lig_mol2: str, out_dir: str) -> None:
        self.prot_pdb = prot_pdb
        self.lig_mol2 = lig_mol2
        self.out_dir = out_dir

    def run(self) -> None:
        prot_pdb: str = self.prot_pdb
        lig_mol2: str = self.lig_mol2
        fl: str = osp.basename(prot_pdb).split("_")[0]

        # generate initial 3D geometries
        tmp_dir: str = "./tmp"
        cmd = f"obabel -imol2 {lig_mol2} -osdf -O {fl}_ligand_2D.sdf --gen2D"
        subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)
        cmd = f"obabel -isdf {fl}_ligand_2D.sdf -osdf -O {fl}_ligand_3D.sdf --gen3D"
        subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)
        cmd = f"obabel {fl}_ligand_3D.sdf -O conformers.sdf --conformer --nconf 10 --score rmsd --writeconformers"
        subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)
        os.makedirs(osp.join(tmp_dir, "conformers"), exist_ok=True)
        cmd = f"python /softwares/Lin_F9_test/split_sdf.py conformers.sdf conformers/"
        subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)

        # generate PDBQT files
        cmd = f"{PREPARE_PROT} -r {prot_pdb} -U nphs_lps -A 'checkhydrogens' -o {fl}_protein.pdbqt"
        subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)
        # a weird bug: prepare_lig only works in the folder where the source file is located 
        lig_pdbqt = osp.abspath(osp.join(tmp_dir, f"{fl}_ligand.pdbqt"))
        cmd = f"{PREPARE_LIG} -l {lig_mol2} -U nphs_lps -A 'checkhydrogens' -o {lig_pdbqt}"
        subprocess.run(cmd, shell=True, check=True, cwd=osp.dirname(lig_mol2))

        # global docking
        os.makedirs(self.out_dir, exist_ok=True)
        confs: List[str] = glob(osp.join(tmp_dir, "conformers", "*.mol2"))
        for conf in confs:
            conf = osp.abspath(conf)
            base: str = osp.basename(conf).split(".mol2")[0]
            out_f: str = osp.abspath(osp.join(self.out_dir, f"{base}.pdb"))
            cmd = f"{LINF9} -r 1a30_protein.pdbqt -l {conf} --autobox_ligand 1a30_ligand.pdbqt --scoring Lin_F9 -o {out_f} "
            subprocess.run(cmd, shell=True, check=True, cwd=tmp_dir)
