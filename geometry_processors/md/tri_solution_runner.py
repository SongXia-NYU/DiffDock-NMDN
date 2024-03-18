import json
import os.path as osp
import os
import subprocess
from time import time
from typing import Dict

from geometry_processors.md.amber_calculator import AmberCalculator, GBSACalculator

DIEL_MAPPER = {"gas": 1.0, "water": 78.5, "octanol": 10.3}


class TriSolutionRunner:
    """
    A wrapper class to run AmberCalculator more efficiently
    """

    def __init__(self, method, pdb_f, root, **calc_kwargs) -> None:
        self.method = method
        self.pdb_f = pdb_f
        self.base = osp.basename(pdb_f).split(".pdb")[0]
        self.run_dir = osp.join(root, self.base)
        self.root = root

        self._calculators = None
        self.calc_kwargs = calc_kwargs
        self._out_files = None
        self._summary_csvs = None
        self._amber_h_folder = None
        self._out_folders = None
        self._csv_folders = None
        self._time_info_folder = None
        self._last_time = time()
        self._time_info = {}

    def run(self):
        if osp.exists(self.run_dir):
            if self.is_finished():
                print(f"{self.base} exists and finished, skipping...")
                return
            else:
                print(f"{self.base} exists but not finished, removing...")
                if osp.exists(self.run_dir):
                    subprocess.run(f"rm -r {self.run_dir}", shell=True, check=True)
        self._run()

    def _run(self):
        os.makedirs(self.run_dir, exist_ok=False)
        self.record_time("mkdirs")
        inited = False
        for phase in self.calculators:
            calc = self.calculators[phase]
            if not inited:
                calc.gen_amber_pdb()
                self.record_time("gen_amber_pdb")
                calc.gen_tleapin()
                self.record_time("gen_tleapin")
                calc.gen_parm_files()
                self.record_time("gen_parm_files")
                inited = True
            calc.gen_mdin()
            self.record_time(f"gen_mdin.{phase}")
            calc.run_amber()
            self.record_time(f"run_amber.{phase}")
            calc.read_output()
            self.record_time(f"read_output.{phase}")
            subprocess.run(f"cd {self.run_dir}; mv {calc.out_file} {self.out_files[phase]}", shell=True, check=True)
            subprocess.run(f"cd {self.run_dir}; mv {calc.summary_csv} {self.summary_csvs[phase]}", shell=True,
                           check=True)
            self.record_time(f"renaming.{phase}")
        calc.del_temp_files()
        self.record_time("del_temp_files")

        subprocess.run(f"cd {self.run_dir}; mv {calc.amber_pdb} {self.amber_h_folder}", shell=True, check=True)
        subprocess.run(f"rm -r '{self.run_dir}'", shell=True, check=True)
        self.record_time("clear_up")

        with open(osp.join(self.time_info_folder, f"{self.base}.json"), "w") as f:
            json.dump(self._time_info, f, indent=2)

    def record_time(self, key):
        dt = time() - self._last_time
        if key in self._time_info:
            self._time_info[key] += dt
        else:
            self._time_info[key] = dt
        self._last_time = time()

    def is_finished(self):
        for phase in self.calculators:
            if not osp.exists(self.out_files[phase]) or not osp.exists(self.summary_csvs[phase]):
                return False
        return True

    @property
    def amber_h_folder(self):
        if self._amber_h_folder is None:
            proposed_folder = osp.join(self.root, "amber_h_pdbs")
            os.makedirs(proposed_folder, exist_ok=True)
            self._amber_h_folder = proposed_folder
        return self._amber_h_folder

    @property
    def time_info_folder(self):
        if self._time_info_folder is None:
            proposed_folder = osp.join(self.root, "time_info")
            os.makedirs(proposed_folder, exist_ok=True)
            self._time_info_folder = proposed_folder
        return self._time_info_folder

    @property
    def out_folders(self):
        if self._out_folders is None:
            result = {}
            for phase in ["gas", "water", "octanol"]:
                proposed_folder = osp.join(self.root, f"logs.{phase}")
                os.makedirs(proposed_folder, exist_ok=True)
                result[phase] = proposed_folder
            self._out_folders = result
        return self._out_folders

    @property
    def csv_folders(self):
        if self._csv_folders is None:
            result = {}
            for phase in ["gas", "water", "octanol"]:
                proposed_folder = osp.join(self.root, f"csvs.{phase}")
                os.makedirs(proposed_folder, exist_ok=True)
                result[phase] = proposed_folder
            self._csv_folders = result
        return self._csv_folders

    @property
    def calculators(self) -> Dict[str, AmberCalculator]:
        if self._calculators is None:
            if self.method.lower() == "gbsa":
                calcs = {}
                for phase in ["gas", "water", "octanol"]:
                    mdin_kwargs = {"extdiel": DIEL_MAPPER[phase], "gbsa": 1}
                    calc = GBSACalculator(self.pdb_f, self.run_dir, mdin_kwargs=mdin_kwargs, **self.calc_kwargs)
                    calcs[phase] = calc
            else:
                raise NotImplemented

            self._calculators = calcs
        return self._calculators

    @property
    def out_files(self):
        if self._out_files is None:
            res = {}
            for phase in self.calculators:
                out_f = osp.basename(self.calculators[phase].out_file).replace(".out", f".{phase}.out")
                out_f = osp.join(self.out_folders[phase], out_f)
                res[phase] = out_f
            self._out_files = res
        return self._out_files

    @property
    def summary_csvs(self):
        if self._summary_csvs is None:
            res = {}
            for phase in self.calculators:
                out_f = osp.basename(self.calculators[phase].summary_csv).replace(".csv", f".{phase}.csv").replace("summary", f"{self.base}")
                out_f = osp.join(self.csv_folders[phase], out_f)
                res[phase] = out_f
            self._summary_csvs = res
        return self._summary_csvs
