import subprocess
import os
import os.path as osp
from glob import glob
import warnings
import pandas as pd

from geometry_processors.md.amber_reader import GBSAReader, PBSAReader

PDB4AMBER = "/share/apps/amber/22.00/openmpi/intel/bin/pdb4amber"
TLEAP = "/share/apps/amber/22.00/openmpi/intel/bin/tleap"
PBSA = "/share/apps/amber/22.00/openmpi/intel/bin/pbsa"
GASE = "/share/apps/amber/22.00/openmpi/intel/bin/mmpbsa_py_energy"
SANDER = "/share/apps/amber/22.00/openmpi/intel/bin/sander"
AMBPDB = "/share/apps/amber/22.00/openmpi/intel/bin/ambpdb"


class AmberCalculator:
    def __init__(self, pdb_f, run_dir, templ_tleap_f, mdin_templ_f, mdin_kwargs=None, rm_pdb=False) -> None:
        self.rm_pdb = rm_pdb
        self.mdin_templ_f = mdin_templ_f
        self.mdin_kwargs = mdin_kwargs if mdin_kwargs else {}
        self.templ_tleap_f = templ_tleap_f
        self.pdb_f = pdb_f
        self.run_dir = run_dir

        self._prmtop = None
        self._inpcrd = None
        self._amber_pdb = None
        self._base_name = None
        self._tleap_in = None
        self._tleap_templ = None
        self._mdin_templ = None
        self._out_file = None
        self._summary_csv = None
        self._mdin = None

        self._normal_termination = None

    def run(self):
        warnings.warn(
            "Please use TriSolutionRunner if you are calculating in multiple solutions. Otherwise ignore this.")
        if osp.exists(self.run_dir):
            if self.normal_termination:
                print(f"{self.run_dir} exists and finished, skipping...")
                return
            else:
                print(f"{self.run_dir} exists but not finished, removing...")
                subprocess.run(f"rm -r {self.run_dir}", shell=True, check=True)
        self._run()

    @property
    def normal_termination(self):
        if self._normal_termination is None:
            if osp.exists(self.summary_csv) and osp.exists(self.out_file):
                self._normal_termination = True
            else:
                self._normal_termination = False
        return self._normal_termination

    def _run(self):
        os.makedirs(self.run_dir, exist_ok=False)

        self.gen_amber_pdb()
        self.gen_tleapin()
        self.gen_parm_files()
        self.gen_mdin()
        self.run_amber()
        self.del_temp_files()
        self.read_output()

    def gen_amber_pdb(self):
        # generate amber
        # amber has some issues dealing with long path names, so I have to use relative path
        cmd = f"cd '{self.run_dir}'; {PDB4AMBER} -i '{self.pdb_f}' -o '{osp.basename(self.amber_pdb)}' --dry "
        subprocess.run(cmd, shell=True, check=True)

    def gen_tleapin(self):
        # generate tleap.in
        with open(self.tleap_in, "w") as f:
            f.write(self.tleap_templ.format(amber_pdb=osp.basename(self.amber_pdb),
                prmtop=osp.basename(self.prmtop),
                inpcrd=osp.basename(self.inpcrd)))

    def gen_parm_files(self):
        # generate parameter files
        cmd = f"cd '{self.run_dir}'; {TLEAP} -f '{osp.basename(self.tleap_in)}'"
        subprocess.run(cmd, shell=True, check=True)

    def gen_mdin(self):
        # generate md.in
        with open(self.mdin, "w") as f:
            f.write(self.mdin_templ.format(**self.mdin_kwargs))

    def run_amber(self):
        raise NotImplementedError

    def del_temp_files(self):
        # delete all but the output file
        cmd = f"rm {osp.join(self.amber_pdb).replace('.pdb', '_*')} '{self.inpcrd}' '{self.prmtop}' '{self.tleap_in}' '{self.mdin}'"
        subprocess.run(cmd, shell=True, check=True)
        cmd = f"cd '{self.run_dir}'; rm leap.log mdinfo "
        subprocess.run(cmd, shell=True, check=True)
        if self.rm_pdb:
            subprocess.run(f"rm '{self.amber_pdb}'", shell=True, check=True)

    def read_output(self):
        raise NotImplementedError

    @property
    def mdin(self):
        if self._mdin is None:
            self._mdin = osp.join(self.run_dir, "md.in")
        return self._mdin

    @property
    def amber_pdb(self):
        if self._amber_pdb is None:
            self._amber_pdb = osp.join(self.run_dir, f"{self.base_name}.amber.H.pdb")
        return self._amber_pdb

    @property
    def base_name(self):
        if self._base_name is None:
            base_name = osp.basename(self.pdb_f).split(".pdb")[0]
            self._base_name = base_name
        return self._base_name

    @property
    def tleap_in(self):
        if self._tleap_in is None:
            self._tleap_in = osp.join(self.run_dir, "tleap.in")
        return self._tleap_in

    @property
    def prmtop(self):
        if self._prmtop is None:
            self._prmtop = osp.join(self.run_dir, f"{self.base_name}.parmtop")
        return self._prmtop

    @property
    def inpcrd(self):
        if self._inpcrd is None:
            self._inpcrd = osp.join(self.run_dir, f"{self.base_name}.inpcrd")
        return self._inpcrd

    @property
    def tleap_templ(self):
        if self._tleap_templ is None:
            with open(self.templ_tleap_f) as f:
                self._tleap_templ = f.read()
        return self._tleap_templ

    @property
    def mdin_templ(self):
        if self._mdin_templ is None:
            with open(self.mdin_templ_f) as f:
                self._mdin_templ = f.read()
            return self._mdin_templ

    @property
    def out_file(self):
        if self._out_file is None:
            self._out_file = osp.join(self.run_dir, f"{self.base_name}.out")
        return self._out_file

    @property
    def summary_csv(self):
        if self._summary_csv is None:
            self._summary_csv = osp.join(self.run_dir, "summary.csv")
        return self._summary_csv


class PBSACalculator(AmberCalculator):
    def __init__(self, pdb_f, run_dir, epsout=80., **kwargs) -> None:
        TEMPL = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/PBSA.tleap.in"
        TEMPL_MIDIN = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/PBSA.in"
        super().__init__(pdb_f, run_dir, TEMPL, TEMPL_MIDIN, {"epsout": epsout}, **kwargs)

    def run_amber(self):
        # run pbsa
        cmd = f"cd '{self.run_dir}'; {PBSA} -i '{osp.basename(self.mdin)}' -o '{osp.basename(self.out_file)}' -p '{osp.basename(self.prmtop)}' -c '{osp.basename(self.inpcrd)}' "
        subprocess.run(cmd, shell=True, check=True)

    def read_output(self):
        reader = PBSAReader(self.out_file)
        out_df = pd.DataFrame(reader.info_dict, index=[self.base_name])
        out_df.to_csv(self.summary_csv, index=True)


class GBSACalculator(AmberCalculator):
    def __init__(self, pdb_f, run_dir, mdin_kwargs: dict = None, **kwargs) -> None:
        TEMPL = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/GBSA.tleap.in"
        TEMPL_MDIN = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/GBSA.in"
        mdin_kwargs_ = {"igb": 8, "extdiel": 78.5, "gbsa": 0}
        if mdin_kwargs is not None:
            mdin_kwargs_.update(mdin_kwargs)
        super().__init__(pdb_f, run_dir, TEMPL, TEMPL_MDIN, mdin_kwargs_, **kwargs)

    def read_output(self):
        reader = GBSAReader(self.out_file)
        out_df = pd.DataFrame(reader.info_dict, index=[self.base_name])
        out_df.to_csv(self.summary_csv, index=True)

    def run_amber(self):
        # run gas energy calculation
        cmd = f"cd '{self.run_dir}'; {SANDER} -i '{osp.basename(self.mdin)}' -O -o '{osp.basename(self.out_file)}' -p '{osp.basename(self.prmtop)}' -c '{osp.basename(self.inpcrd)}' "
        subprocess.run(cmd, shell=True, check=True)
        # cmd = f"cd '{self.run_dir}'; {AMBPDB} -p '{osp.basename(self.prmtop)}' -c restrt > ambpdb.pdb"
        # subprocess.run(cmd, shell=True, check=True)

    def del_temp_files(self):
        subprocess.run(f"cd '{self.run_dir}'; rm restrt ", shell=True, check=True)
        return super().del_temp_files()


class GBSAOptCalculator(AmberCalculator):
    def __init__(self, pdb_f, run_dir, mdin_kwargs: dict = None, **kwargs) -> None:
        TEMPL = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/GBSA.tleap.in"
        TEMPL_MDIN = "/scratch/sx801/scripts/Mol3DGenerator/configs/amber/GBSA_OPT.in"
        mdin_kwargs_ = {"igb": 8, "extdiel": 78.5, "gbsa": 0}
        if mdin_kwargs is not None:
            mdin_kwargs_.update(mdin_kwargs)
        super().__init__(pdb_f, run_dir, TEMPL, TEMPL_MDIN, mdin_kwargs_, **kwargs)

    def read_output(self):
        reader = GBSAReader(self.out_file)
        out_df = pd.DataFrame(reader.info_dict, index=[self.base_name])
        out_df.to_csv(self.summary_csv, index=True)

    def run_amber(self):
        # run gas energy calculation
        cmd = f"{SANDER} -i '{osp.basename(self.mdin)}' -O -o '{osp.basename(self.out_file)}' -p '{osp.basename(self.prmtop)}' -c '{osp.basename(self.inpcrd)}' "
        subprocess.run(cmd, shell=True, check=True, cwd=self.run_dir)
        cmd = f"{AMBPDB} -p '{osp.basename(self.prmtop)}' -c restrt > {self.base_name}.opt.pdb"
        subprocess.run(cmd, shell=True, check=True, cwd=self.run_dir)

    def del_temp_files(self):
        subprocess.run(f"rm restrt {self.amber_pdb}", shell=True, check=True, cwd=self.run_dir)
        return super().del_temp_files()
