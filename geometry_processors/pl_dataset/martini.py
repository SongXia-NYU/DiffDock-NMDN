import json
from multiprocessing import Pool
import os
import subprocess
import prody
from prody import parsePDB, writePDB
import os.path as osp
from ase import Atom
from time import time
import tempfile
from geometry_processors.md.amber_calculator import PDB4AMBER

from geometry_processors.misc import solv_num_workers

STD_OUT = subprocess.DEVNULL


DSSP = "/share/apps/dssp/3.1.4/intel/bin/mkdssp"
MARTINI_PATH = "/scratch/sx801/scripts/martini_3.0.b.3.2"
MARTINIE=f"{MARTINI_PATH}/martinize"
MARTINI_FF = f"{MARTINI_PATH}/martini303v.partition"

"""
This file contains the script to generate the Martini Coarse Grained structures.
The script CANNOT do MD simulation.
To perform Martini MD simulation, check martini_md.py in the same folder.
"""

class SingleMartiniCalculator:
    def __init__(self, f_in, f_out, dssp_file, top_out, add_cap, amber_h_pdb=None) -> None:
        self.f_in = f_in
        self.f_out = f_out
        self.dssp_file = dssp_file
        self.top_out = top_out
        self.add_cap = add_cap
        self.amber_h_pdb = amber_h_pdb

    def run(self):
        if osp.exists(self.f_out) and osp.exists(self.top_out):
            return
        with tempfile.TemporaryDirectory() as tempdir:
            inpdb = self.f_in
            if self.amber_h_pdb is not None:
                cmd = f"cd '{tempdir}'; {PDB4AMBER} -i '{inpdb}' -o '{self.amber_h_pdb}' --reduce "
                subprocess.run(cmd, shell=True, check=True)
                inpdb = self.amber_h_pdb
            subprocess.run(f"cd {tempdir}; {DSSP} -i {inpdb} -o {self.dssp_file}", shell=True, check=True, stdout=STD_OUT, stderr=STD_OUT)
            martini_cmd = f"cd {tempdir}; {MARTINIE} -f {inpdb} -ff {MARTINI_FF} -x {self.f_out} -o {self.top_out} -ss {self.dssp_file} -elastic"
            subprocess.run(martini_cmd, shell=True, check=True, stdout=STD_OUT, stderr=STD_OUT)

        if self.add_cap:
            src_pdb = parsePDB(self.f_in)
            
            massed_sum = None
            total_mass = 0.
            for atom in src_pdb.select("resname ACE"):
                # all hydrogens in ACE are non-polar, therefore they are all ignored
                if atom.getName().startswith("H"):
                    continue
                ele_sym = atom.getName()[0]
                ele = Atom(ele_sym)
                mass = ele.mass
                if massed_sum is None:
                    massed_sum = atom.getCoords() * mass
                else:
                    massed_sum += atom.getCoords() * mass
                total_mass += mass
            avg_coord = massed_sum / total_mass
            ace_bead = atom.copy()
            ace_bead.setCoords(avg_coord.reshape(-1, 3))
            ace_bead.setNames("BB")

            massed_sum = None
            total_mass = 0.
            for atom in src_pdb.select("resname NME"):
                # only the hydrogen on nitrogen is used to calculate the center of mass
                if atom.getName().startswith("H") and atom.getName() != "H":
                    continue
                ele_sym = atom.getName()[0]
                ele = Atom(ele_sym)
                mass = ele.mass
                if massed_sum is None:
                    massed_sum = atom.getCoords() * mass
                else:
                    massed_sum += atom.getCoords() * mass
                total_mass += mass
            avg_coord = massed_sum / total_mass
            nme_bead = atom.copy()
            nme_bead.setCoords(avg_coord.reshape(-1, 3))
            nme_bead.setNames("BB")

            main_beads = parsePDB(self.f_out)
            main_beads.delFlags("pdbter")
            cap_martini = ace_bead + main_beads + nme_bead
            cap_martini.setTitle("capped martini")
            writePDB(self.f_out, cap_martini)

    def cleanup(self):
        for f in [self.f_out, self.dssp_file, self.top_out]:
            if osp.exists(f):
                os.remove(f)

def process_one_dssp(args):
    f_in = args[0]
    calculator = SingleMartiniCalculator(*args)
    try:
        calculator.run()
    except Exception as e:
        calculator.cleanup()
        return {"in": f_in, "error": str(e)}


class BatchMartiniCalculator:
    def __init__(self, src_pdbs, save_root="./martini", add_cap=True, mp=True, amber_h=False) -> None:
        self.src_pdbs = src_pdbs
        self.save_root = save_root
        self.add_cap = add_cap
        self.amber_h = amber_h
        self.mp = mp

        self._mp_args = None

    @property
    def mp_args(self):
        if self._mp_args is None:
            mp_args = []
            os.makedirs(osp.join(self.save_root, "pdb"), exist_ok=True)
            os.makedirs(osp.join(self.save_root, "top"), exist_ok=True)
            os.makedirs(osp.join(self.save_root, "dssp"), exist_ok=True)
            if self.amber_h:
                os.makedirs(osp.join(self.save_root, "amber_h"), exist_ok=True)

            for f in self.src_pdbs:
                out_prefix = osp.basename(f).split(".pdb")[0]
                out_f = osp.join(self.save_root, "pdb", f"{out_prefix}.martini.pdb")
                out_top = osp.join(self.save_root, "top", f"{out_prefix}.martini.top")
                out_dssp = osp.join(self.save_root, "dssp", f"{out_prefix}.martini.top")
                if self.amber_h:
                    amber_h_pdb = osp.join(self.save_root, "amber_h", f"{out_prefix}.amber.pdb")
                else:
                    amber_h_pdb = None
                if osp.exists(out_f) and osp.exists(out_top):
                    continue
                mp_args.append((f, out_f, out_dssp, out_top, self.add_cap, amber_h_pdb))
            self._mp_args = mp_args
        return self._mp_args

    def run(self):
        tik = time()
        n_cpu_avail, n_cpu, num_workers = solv_num_workers()
        with Pool(num_workers) as p:
            errors = p.map(process_one_dssp, self.mp_args, chunksize=10)
        errors = [e for e in errors if e]
        tok = time()
        print("delta time : ", tok-tik)
        logs = {"number": len(self.src_pdbs), "dt": tok-tik, "errors": errors}
        with open(osp.join(self.save_root, "info.json"), "w") as f:
            json.dump(logs, f, indent=2)


def preprocess4martini(f_in, f_out):
    # renumber the residues and add missing heavy atoms
    out_base = osp.basename(f_out).split('.')[0]
    f_renum = osp.join(osp.dirname(f_out), f"{out_base}.renum.pdb")
    atoms = parsePDB(f_in)
    # print(atoms.getResnames()[:20])
    # print(atoms.getResnums()[:20])
    prev_res_name = None
    res_id = 1
    prev_origin_id = None
    for atom in atoms:
        this_res = atom.getResname()
        this_origin_id = atom.getResnum()
        if prev_res_name and (this_res != prev_res_name or this_origin_id != prev_origin_id):
            res_id += 1
        atom.setResnum(res_id)
        prev_res_name = this_res
        prev_origin_id = this_origin_id
    writePDB(f_renum, atoms)

    subprocess.run(f"pdbfixer {f_renum} --add-atoms=all --output {f_out}", shell=True, check=True)


def mp_preprocess4martini(args):
    f_in, f_out = args
    try:
        preprocess4martini(f_in, f_out)
    except Exception as e:
        return {"in": f_in, "out": f_out, "error": str(e)}
