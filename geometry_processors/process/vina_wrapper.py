from tempfile import TemporaryDirectory
import os.path as osp
import subprocess

SCRIPT_ROOT = "/softwares"

PREPARE_PROT = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_receptor"
PREPARE_LIG = f"{SCRIPT_ROOT}/ADFRsuite_x86_64Linux_1.0/bin/prepare_ligand"
VINA = "/softwares/Lin_F9_test/smina.static"

class VinaScoreCalculator:
    def __init__(self, tempdir: TemporaryDirectory, scores=None) -> None:
        # TODO: vinardo, ad4_scoring
        self.tempdir: TemporaryDirectory = tempdir
        self.scores = scores

    def compute_score(self, prot: str, lig: str) -> float:
        prot_pdbqt = osp.join(osp.join(self.tempdir.name, osp.basename(prot).replace(".pdb", ".pdbqt").replace(" ", "")))
        if not osp.exists(prot_pdbqt):
            cmd = f"{PREPARE_PROT} -r '{prot}' -U nphs_lps -A 'checkhydrogens' -o {osp.basename(prot_pdbqt)}"
            subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name)
        lig_ext = lig.split(".")[-1]
        lig_pdbqt = osp.join(osp.join(self.tempdir.name, ".".join(osp.basename(lig).split(".")[:-1]).replace(" ", "")+".pdbqt"))
        cmd = f"obabel -i{lig_ext} '{lig}' -opdbqt -O {lig_pdbqt} -h"
        subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name)

        score_mapper = {}
        scores = self.scores if self.scores is not None else ["vina", "ad4_scoring", "vinardo"]
        for score_name in scores:
            cmd = f"{VINA} -r {prot_pdbqt} -l {lig_pdbqt} --scoring {score_name} --local_only "
            res = subprocess.run(cmd, shell=True, check=True, cwd=self.tempdir.name, capture_output=True, text=True)
            for line in res.stdout.split("\n"):
                if line.startswith("Affinity:"):
                    score_mapper[score_name] = float(line.split()[1])
                    break
        return score_mapper

