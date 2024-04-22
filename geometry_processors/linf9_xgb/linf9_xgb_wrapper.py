import os
import os.path as osp
from tempfile import TemporaryDirectory
from geometry_processors.pl_dataset.linf9_local_optimizer import LinF9LocalOptimizer
from runXGB import run_XGB
import subprocess


def xgb_wrapper(prot: str, lig: str, protdir: str, addhs=False, linf9_only=False):
    lig_ifmt = lig.split(".")[-1]
    lig_id = osp.basename(lig).split(f".{lig_ifmt}")[0]
    lig_id = "".join(lig_id.split(" "))
    with TemporaryDirectory() as temp_dir:
        cwd = os.getcwd()
        # goto a temp_dir to avoid generation of temp file in the current directory
        os.chdir(temp_dir)
        if addhs:
            cmd = f"/ext3/miniconda3/bin/obabel -i{lig_ifmt} '{lig}' -osdf -O {lig_id}.sdf -h"
            subprocess.run(cmd, shell=True, check=True, cwd=temp_dir)
            lig = osp.join(temp_dir, f"{lig_id}.sdf")
        try:
            lig_opt = osp.join(temp_dir, osp.basename(lig).replace(".sdf", ".pdb"))
            opt = LinF9LocalOptimizer(ligand_sdf=lig, protein_pdb=prot, ligand_linf9_opt=lig_opt)
            opt.run(protdir)
            res = run_XGB(prot, lig_opt, temp_dir, protdir, linf9_only=linf9_only)
        except Exception as e:
            # raise e
            os.chdir(cwd)
            print(f"ERROR proc {lig_id}: {e}")
            
            return None
    os.chdir(cwd)
    if res is None:
        return None

    assert len(res) == 2, res
    xgb_score, linf9_score = res
    res = {"lig_id": lig_id, "xgb_score": xgb_score, "linf9_score": linf9_score}
    return res
