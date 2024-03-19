import os
import os.path as osp
from tempfile import TemporaryDirectory
from runXGB import run_XGB
import subprocess


def xgb_wrapper(prot: str, lig: str, protdir: str):
    assert lig.endswith(".sdf")
    with TemporaryDirectory() as temp_dir:
        cwd = os.getcwd()
        # goto a temp_dir to avoid generation of temp file in the current directory
        os.chdir(temp_dir)
        cmd = f"/ext3/miniconda3/bin/obabel -isdf {lig} -osdf -O {osp.basename(lig)} -h"
        subprocess.run(cmd, shell=True, check=True, cwd=temp_dir)
        lig = osp.join(temp_dir, osp.basename(lig))
        try:
            res = run_XGB(prot, lig, temp_dir, protdir)
        except Exception as e:
            os.chdir(cwd)
            raise e
            print(f"ERROR proc {osp.basename(lig)}: {e}")
            
            return None
    os.chdir(cwd)
    if res is None:
        return None

    assert len(res) == 2, res
    xgb_score, linf9_score = res
    lig_id = osp.basename(lig).split(".pdb")[0]
    res = {"lig_id": lig_id, "xgb_score": xgb_score, "linf9_score": linf9_score}
    return res
