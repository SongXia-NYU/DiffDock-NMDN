from tempfile import TemporaryDirectory
import os.path as osp
import subprocess

SCRIPT_ROOT = "/softwares"

PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor"
PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_ligand"
VINA = "/scratch/sx801/softwares/autodock_vina_1_1_2_linux_x86/bin/vina"

class VinaScoreCalculator:
    def __init__(self, tempdir: TemporaryDirectory) -> None:
        self.tempdir: TemporaryDirectory = tempdir

    def compute_score(self, prot: str, lig: str) -> float:
        prot_pdbqt = osp.join(osp.join(self.tempdir.name, osp.basename(prot).replace(".pdb", ".pdbqt").replace(" ", "")))
        if not osp.exists(prot_pdbqt):
            cmd = f"{PREPARE_PROT} -r '{prot}' -U nphs_lps -A 'checkhydrogens' -o {osp.basename(prot_pdbqt)}"
            subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name)
        lig_ext = lig.split(".")[-1]
        lig_pdbqt = osp.join(osp.join(self.tempdir.name, ".".join(osp.basename(lig).split(".")[:-1]).replace(" ", "")))
        cmd = f"obabel -i{lig_ext} '{lig}' -opdbqt -O {lig_pdbqt} -h"
        subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name)

        cmd = f"{VINA} --receptor {prot_pdbqt} --ligand {lig_pdbqt} --score_only"
        res = subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name, capture_output=True, text=True)
        for line in res.stdout.split("\n"):
            if line.startswith("Affinity:"):
                return float(line.split()[1])

