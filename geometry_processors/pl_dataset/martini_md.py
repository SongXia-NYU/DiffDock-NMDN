from functools import partial
import os
import subprocess
from typing import List, Union
import os.path as osp
import glob
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from prody import parsePDB, calcRMSD

from geometry_processors.pl_dataset.martini import DSSP, MARTINI_PATH, MARTINIE, MARTINI_FF

from geometry_processors.misc import solv_num_workers

BSSC = f"{MARTINI_PATH}/bbsc.sh"
VMD_BIN = "/share/apps/vmd/1.9.3/bin"
INSANE = f"{MARTINI_PATH}/insane.py"
GMX = "/share/apps/gromacs/2020.4/openmpi/intel/bin/gmx_mpi"
ENERGY_MDP = f"{MARTINI_PATH}/martini_v3.x_energy.mdp"
ENERGY_MIN_MDP = f"{MARTINI_PATH}/martini_v3.x_energy_min.mdp"

STD_OUT = subprocess.DEVNULL

TOP_HEAD = \
f'''#include "{MARTINI_PATH}/martini_v3.0.b.3.2.itp" 
#include "{MARTINI_PATH}/martini_v3.0_ions.itp" 
#include "{MARTINI_PATH}/martini_v3.0_solvents.itp"
'''

def _single_martini_runner(src_pdb, save_root, **kwargs):
    calculator = SingleMartiniMDCalculator(save_root, src_pdb, **kwargs)
    try:
        calculator.run()
    except Exception as e:
        print(e)
        calculator.remove_dir()
        # raise e

class BatchMartiniMDCalculator:
    """
    Running Martini MD using multiple cpus
    """
    def __init__(self, save_root: str, src_pdbs: Union[str, List[str]], energy_min=False) -> None:
        self.save_root = save_root
        # if you want to run energy minimization or not. if not, single-point energy is calculated.
        self.energy_min = energy_min

        if isinstance(src_pdbs, str):
            src_pdbs = glob.glob(src_pdbs)
        self.src_pdbs = src_pdbs

    def run(self):
        run_func = partial(_single_martini_runner, save_root=self.save_root, energy_min=self.energy_min)
        __, __, num_workers = solv_num_workers()
        process_map(run_func, self.src_pdbs, max_workers=num_workers+4, chunksize=10)

    def run_sp(self):
        run_func = partial(_single_martini_runner, save_root=self.save_root, energy_min=self.energy_min)
        for src_pdb in tqdm(self.src_pdbs):
            run_func(src_pdb)

class SingleMartiniMDCalculator:
    """
    Calculate the energy of protein in vacuum using Martini Force Field. 
    """
    def __init__(self, save_root, src_pdb, energy_min=False) -> None:
        self.save_root = save_root
        self.src_pdb = src_pdb
        # if you want to run energy minimization or not. if not, single-point energy is calculated.
        self.energy_min = energy_min
        
        self.pdb_base = osp.basename(src_pdb).split(".pdb")[0]
        self.work_dir = osp.abspath(osp.join(self.save_root, self.pdb_base))

        # secondary structure
        self.ssd_name = f"{self.pdb_base}.ssd"
        # coarse-grained pdb structure and topology file
        self.cg_pdb_name = f"{self.pdb_base}-CG.pdb"
        self.cg_top_name = f"{self.pdb_base}-CG.top"
        self.cg_top_name_correct = f"{self.pdb_base}-CG-correct.top"
        # itp file
        self.itp_name = "Protein_A.itp"
        # Gromacs file
        self.gro_name = f"{self.pdb_base}.gro"

    def run(self):
        if osp.exists(osp.join(self.work_dir, "md.log")):
            print(f"{self.work_dir} finishes, exiting...")
            return
        os.makedirs(self.work_dir, exist_ok=True)
        # the running script is based on the MARTINI 3 tutorial: http://cgmartini.nl/index.php/martini-3-tutorials/small-molecule-binding
        # calculate the secondary structure of the protein needed by MARTINI
        self.run_dssp()
        # 1B. Generate Martini topology and coordinates
        self.run_martinize()
        # 1C. Add dihedral corrections to Martini topology
        self.run_bssc()
        # 2A. Solvate the protein
        # in addition to running the insane.py script, the method also correct the include files as well as the solvent molecule counts
        self.run_insane()
        # 2C. Start the simulation
        self.run_grompp()
        self.run_md()
        self.post_md()

    def run_dssp(self):
        cmd = f"cd {self.work_dir}; {DSSP} -i {self.src_pdb} -o {self.ssd_name}"
        subprocess.run(cmd, shell=True, check=True, stdout=STD_OUT, stderr=subprocess.STDOUT)
    
    def run_martinize(self):
        cmd = f"cd {self.work_dir}; {MARTINIE} -f {self.src_pdb} -ff {MARTINI_FF} -x {self.cg_pdb_name} -o {self.cg_top_name} -ss {self.ssd_name} -elastic"
        subprocess.run(cmd, shell=True, check=True, stdout=STD_OUT, stderr=subprocess.STDOUT)

    def run_bssc(self):
        cmd = f"cd {self.work_dir}; PATH={VMD_BIN}:$PATH bash {BSSC} {self.itp_name} {self.src_pdb}"
        subprocess.run(cmd, shell=True, check=True, stdout=STD_OUT, stderr=subprocess.STDOUT)

    def run_insane(self):
        cmd = f"cd {self.work_dir}; python {INSANE} -f {self.cg_pdb_name} -o {self.gro_name} -pbc cubic -box 10,10,10 "
        insane_output = subprocess.run(cmd, shell=True, check=True, capture_output=True)
        count_info_raw = insane_output.stderr.decode().split("\n")
        count_info_raw.reverse()

        with open(osp.join(self.work_dir, self.cg_top_name_correct), "w") as f_out:
            with open(osp.join(self.work_dir, self.cg_top_name)) as f_in:
                for i, line in enumerate(f_in.readlines()):
                    if i == 0:
                        f_out.write(TOP_HEAD)
                    else:
                        f_out.write(line)
            f_out.write("\n")
            for count_line in count_info_raw:
                if count_line == "":
                    continue
                if len(count_line.split()) != 2:
                    break
                f_out.write(count_line.replace("NA+", "TNA").replace("CL-", "TCL") + "\n")

    def run_grompp(self):
        MDP = ENERGY_MIN_MDP if self.energy_min else ENERGY_MDP
        cmd = ["gmx_mpi", "grompp", "-p", self.cg_top_name_correct, "-f", MDP, "-c", self.gro_name, "-maxwarn", "10"]
        subprocess.run(cmd, check=True, cwd=self.work_dir, stdout=STD_OUT, stderr=subprocess.STDOUT)

    def run_md(self):
        cmd = ["gmx_mpi", "mdrun", "-c", "CG-em.gro"]
        subprocess.run(cmd, check=True, cwd=self.work_dir, stdout=STD_OUT, stderr=subprocess.STDOUT)

    def post_md(self):
        if not self.energy_min:
            return
        # read the minimization binary file into a csv file
        potential_gen = "printf 'Potential\n0\n' | gmx_mpi energy -f ener.edr -o potential.xvg -xvg none"
        subprocess.run(potential_gen, shell=True, check=True, cwd=self.work_dir, stdout=STD_OUT, stderr=subprocess.STDOUT)
        # convert the minimized structure back to pdb format
        pdb_gen = "printf '0\n' | gmx_mpi trjconv -s topol.tpr -f traj.trr -o minimized.pdb -pbc whole"
        subprocess.run(pdb_gen, shell=True, check=True, cwd=self.work_dir, stdout=STD_OUT, stderr=subprocess.STDOUT)

        # remove large files to reduce disk usage
        rm_command = f"rm ener.edr md.log Protein_A.itp topol.tpr"
        subprocess.run(rm_command, shell=True, check=True, cwd=self.work_dir, stdout=STD_OUT, stderr=subprocess.STDOUT)

    def remove_dir(self):
        cmd = f"rm -r {self.work_dir}"
        subprocess.run(cmd, shell=True, check=True, stdout=STD_OUT, stderr=subprocess.STDOUT, cwd=self.work_dir)
        

class SingleMartiniReader:
    def __init__(self) -> None:
        pass

    @property
    def prop_df(self):
        raise NotImplementedError

class SingleMartiniMDReader(SingleMartiniReader):
    """
    Read Martini MD output file
    """
    def __init__(self, log_file) -> None:
        super().__init__()
        self.log_file = log_file

        self._log_lines = None
        self._num_lines = None
        self._normal_terminated = None

        self._prop_dict =None
        self._prop_df = None

    @property
    def prop_df(self):
        if self._prop_df is None:
            self._prop_df = pd.DataFrame(self.prop_dict, index=[0]).set_index("file_handle")
        return self._prop_df

    @property
    def prop_dict(self):
        if self._prop_dict is None:
            self.scan_lines()
        return self._prop_dict

    def scan_lines(self):
        self._prop_dict = {}
        file_handle = osp.basename(osp.dirname(self.log_file))
        self._prop_dict["file_handle"] = file_handle

        if not self.normal_terminated:
            return

        start_record = False
        for i, line in enumerate(self.log_lines):
            if "<====  A V E R A G E S  ====>" in line:
                start_record = True
            if not start_record:
                continue
            
            if "Energies (kJ/mol)" in line:
                PARTIALS = set(["(SR)", "En.", "Energy", "(bar)", "Dih.", "rmsd", "Pot."])
                while True:
                    label_line = self.log_lines[i + 1]
                    data_line = self.log_lines[i + 2]
                    labels_raw = label_line.split()
                    labels_proc = []
                    if len(labels_raw) == 0:
                        break
                    for label_raw in labels_raw:
                        if label_raw in PARTIALS:
                            labels_proc[-1] += f" {label_raw}"
                        else:
                            labels_proc.append(label_raw)
                    data_float = [float(d) for d in data_line.split()]
                    assert len(labels_proc) == len(data_float), f"{labels_proc} {data_float}"
                    for label, data in zip(labels_proc, data_float):
                        self._prop_dict[label] = data
                    i += 2

    @property
    def normal_terminated(self):
        if self._normal_terminated is None:
            self._normal_terminated = False
            for i in range(self.num_lines):
                this_line = self.log_lines[self.num_lines - i - 1]
                if this_line.startswith("Finished mdrun on"):
                    self._normal_terminated = True
        return self._normal_terminated

    @property
    def log_lines(self):
        if self._log_lines is None:
            with open(self.log_file) as f:
                self._log_lines = f.readlines()
            self._num_lines = len(self._log_lines)
        return self._log_lines
    
    @property
    def num_lines(self):
        if self._num_lines is None:
            __ = self.log_lines
        return self._num_lines


def _reader_wrapper(log_file):
    reader = SingleMartiniMDReader(log_file)
    return reader.prop_df

class BatchMartiniMDReader:
    def __init__(self, log_files: Union[List, str], save_root=".") -> None:
        if isinstance(log_files, str):
            log_files = glob.glob(log_files)
        self.ds_name = osp.basename(osp.dirname(osp.dirname(log_files[0])))
        self.log_files = log_files
        self.save_root = save_root

    def run(self, mp=True):
        if mp:
            prop_dfs = self._run_mp()
        else:
            prop_dfs = self._run_sp()
        prop_dfs = pd.concat(prop_dfs)
        prop_dfs.to_csv(osp.join(self.save_root, self.ds_name+".csv"))

    def _run_mp(self):
        return process_map(_reader_wrapper, self.log_files, max_workers=12, chunksize=10)
    
    def _run_sp(self):
        return [_reader_wrapper(log_file) for log_file in tqdm(self.log_files)]


class SingleMartiniEMReader(SingleMartiniReader):
    """
    Read the result folder of Martini Energy Minimization
    """
    def __init__(self, result_folder, clean=False) -> None:
        super().__init__()
        self.result_folder = result_folder
        # clean up some backup files that is generated during Gromacs simulation
        self.clean = clean

        self._prop_df = None
        self._prop_dict = None
        self._potential_df = None

    @property
    def prop_df(self):
        if self._prop_df is None:
            if self.clean:
                self.do_clean()
            self._prop_df = pd.DataFrame(self.prop_dict, index=[0]).set_index("file_handle")
        return self._prop_df

    @property
    def prop_dict(self):
        if self._prop_dict is None:
            res = {}
            file_handle = osp.basename(self.result_folder)
            res["file_handle"] = file_handle
            res["num_iter"] = int(self.potential_df.iloc[-1][0])
            res["potential"] = self.potential_df.iloc[-1][1]

            p_b4 = parsePDB(osp.join(self.result_folder, f"{file_handle}-CG.pdb"))
            p_after = parsePDB(osp.join(self.result_folder, "minimized.pdb"))
            rmsd = calcRMSD(p_b4, p_after)
            res["RMSD"] = rmsd

            n_bead_before = p_b4.numAtoms()
            n_bead_after = p_after.numAtoms()
            assert n_bead_before == n_bead_after, f"{n_bead_before}, {n_bead_after}"
            res["n_bead"] = n_bead_before
            n_aa_before = p_b4.numResidues()
            n_aa_after = p_after.numResidues()
            assert n_aa_before == n_aa_after, f"{n_aa_before}, {n_aa_after}"
            res["n_aa"] = n_aa_before

            self._prop_dict = res
        return self._prop_dict

    @property
    def potential_df(self):
        if self._potential_df is None:
            self._potential_df = pd.read_csv(osp.join(self.result_folder, "potential.xvg"), sep="\s+", header=None)
        return self._potential_df

    def do_clean(self):
        for f in glob.glob(osp.join(self.result_folder, "#*#")):
            # print(f"removing: {f}")
            os.remove(f)

def _reader_wrapper_em(res_folder):
    reader = SingleMartiniEMReader(res_folder, clean=True)
    try:
        return reader.prop_df
    except Exception as e:
        print(e)
        return None

class BatchMartiniEMReader:
    def __init__(self, res_folders: Union[List, str], save_root=".") -> None:
        if isinstance(res_folders, str):
            res_folders = glob.glob(res_folders)
        self.ds_name = osp.basename(osp.dirname(res_folders[0]))
        self.res_folders = res_folders
        self.save_root = save_root

    def run(self, mp=True):
        if mp:
            prop_dfs = self._run_mp()
        else:
            prop_dfs = self._run_sp()
        prop_dfs = [df for df in prop_dfs if df is not None]
        prop_dfs = pd.concat(prop_dfs)
        prop_dfs.to_csv(osp.join(self.save_root, self.ds_name+".csv"))

    def _run_mp(self):
        return process_map(_reader_wrapper_em, self.res_folders, max_workers=12, chunksize=10)
    
    def _run_sp(self):
        return [_reader_wrapper_em(log_file) for log_file in tqdm(self.res_folders)]

if __name__ == "__main__":
    reader = SingleMartiniEMReader("/vast/sx801/MartiniMD/AF-SwissProt500-MartiniOPT/AF-A0A009IHW8-F1-model_v3", clean=True)
    reader.do_clean()
    print(reader.prop_df)
