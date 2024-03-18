import argparse
import copy
import glob
import os.path as osp
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from ase.units import Hartree, eV
from rdkit.Chem import SDMolSupplier, MolToSmiles, MolFromMolFile
from tqdm import tqdm
from mendeleev import get_all_elements


def read_log(software):
    from geometry_processors.orca.read_orca_log import OrcaLog
    LogClass = Gauss16Log if software == "gauss" else OrcaLog
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_list", type=str)
    parser.add_argument("--log_sdf_list", type=str)
    parser.add_argument("--log_folder", default=None)
    parser.add_argument("--log_sdf_folder", default=None)
    parser.add_argument("--out_path")
    args = parser.parse_args()
    args_cp = deepcopy(args)
    if args.log_folder is not None:
        log_list = glob.glob(args.log_folder)
        log_list.sort()
        setattr(args, "log_list", log_list)
        log_sdf_list = glob.glob(args.log_sdf_folder)
        log_sdf_list.sort()
        setattr(args, "log_sdf_list", log_sdf_list)
    else:
        for prop_name in ["log_list", "log_sdf_list"]:
            with open(getattr(args_cp, prop_name)) as f:
                setattr(args, prop_name, f.read().split())

    result = pd.DataFrame()
    for log, log_sdf in tqdm(zip(args.log_list, args.log_sdf_list), total=len(args.log_list)):
        try:
            gauss_log = LogClass(log, log_sdf)
            result = pd.concat([result, gauss_log.get_data_frame()])
            # result = result.append(gauss_log.prop_dict, ignore_index=True)
        except Exception as e:
            print(f"something is wrong with {log}")
            print(e)

    result.to_csv(args.out_path)


# element_dict = {e.symbol: e.atomic_number for e in get_all_elements()}
# element_periodic = {e.symbol: [e.period, e.group.group_id if e.group is not None else None]
#                     for e in get_all_elements()}


class Gauss16Log:
    def __init__(self, log_path, log_sdf, gauss_version: int = 16, supress_warning=False):
        """
        Extract information from Gaussian 16 log files OR from SDF files. In the later case, qm_sdf, dipole and
        prop_dict_raw must not be None
        """
        self.log_sdf = log_sdf
        self.gauss_version = gauss_version
        self.log_path = log_path

        self.hartree2ev = Hartree / eV
        self.base_name = osp.basename(log_path).split(".")[0] if log_path else None
        base_name1 = osp.basename(log_sdf).split(".")[0] if log_sdf else None
        if self.base_name is None:
            self.base_name = base_name1
        if self.base_name is not None and base_name1 is not None:
            assert self.base_name == base_name1
        self.dir = osp.dirname(log_path) if log_path else None

        self._log_lines = None
        self._log_lines_rev = None
        self._normal_termination = None
        self._mol = None
        self._coordinates = None
        self._n_atoms = None
        self._elements = None
        self._dipole = None
        self._prop_dict_raw = None
        self._charges_mulliken = None
        self._reference_u0 = None
        self._reference = None

        if not supress_warning and not self.normal_termination:
            print("{} did not terminate normally!!".format(log_path))
            return
        self.supress_warning = supress_warning
        # os.system("obabel -ig16 {} -osdf -O {}".format(log_path, qm_sdf))

    @property
    def reference(self):
        if self._reference is not None: return self._reference
        self._reference = np.load(osp.join(osp.dirname(__file__), "atomref.B3LYP_631Gd.10As.npz"))["atom_ref"]
        return self._reference

    @property
    def prop_dict(self):
        """
        Migrated from Jianing's Frag20_prepare:
        https://github.com/jenniening/Frag20_prepare/blob/master/DataGen/prepare_data.py
        :return:
        """
        if self._prop_dict_raw is None:
            self._prop_dict_raw = {"sample_id": self.base_name}
            if self.normal_termination:
                listeners = self.get_listeners()
                for i, line in enumerate(self.log_lines_rev):
                    if not listeners:
                        break
                    next_listeners = []
                    for listener in listeners:
                        if not listener(i, line, self.log_lines_rev, self._prop_dict_raw, hartree2ev=self.hartree2ev):
                            next_listeners.append(listener)
                    listeners = next_listeners

                """ Get properties for each molecule, and convert properties in Hartree unit into eV unit """
                if "U0(eV)" in self._prop_dict_raw.keys():
                    self._prop_dict_raw["U0_atom(eV)"] = (self._prop_dict_raw["U0(eV)"] - self.reference_u0)
                if "E(eV)" in self._prop_dict_raw.keys():
                    self._prop_dict_raw["E_atom(eV)"] = (self._prop_dict_raw["E(eV)"] - self.reference_u0)
                if "U(eV)" in self._prop_dict_raw.keys():
                    reference_total_U = np.sum([self.reference[i][2] for i in self.elements])
                    self._prop_dict_raw["U_atom(eV)"] = (self._prop_dict_raw["U(eV)"] - reference_total_U)
                if "H(eV)" in self._prop_dict_raw.keys():
                    reference_total_H = np.sum([self.reference[i][3] for i in self.elements])
                    self._prop_dict_raw["H_atom(eV)"] = (self._prop_dict_raw["H(eV)"] - reference_total_H)
                if "G(eV)" in self._prop_dict_raw.keys():
                    reference_total_G = np.sum([self.reference[i][4] for i in self.elements])
                    self._prop_dict_raw["G_atom(eV)"] = (self._prop_dict_raw["G(eV)"] - reference_total_G)

                self._prop_dict_raw["smiles"] = MolToSmiles(self.mol, allHsExplicit=False)
        return self._prop_dict_raw

    def get_listeners(self):
        return Gauss16LogListeners.get_all_listeners()

    @property
    def reference_u0(self):
        if self._reference_u0 is None:
            self._reference_u0 = np.sum([self.reference[i][1] for i in self.elements])
        return self._reference_u0

    @property
    def log_lines(self):
        if self._log_lines is None:
            self._log_lines = open(self.log_path).readlines()
        return self._log_lines

    @property
    def log_lines_rev(self):
        if self._log_lines_rev is None:
            tmp = deepcopy(self.log_lines)
            tmp.reverse()
            self._log_lines_rev = tmp
        return self._log_lines_rev

    @property
    def mol(self):
        if self._mol is None:
            self._mol = MolFromMolFile(self.log_sdf, removeHs=False, strictParsing=False)
        return self._mol

    @property
    def n_atoms(self):
        if self._n_atoms is None:
            self._n_atoms = self.mol.GetNumAtoms()
        return self._n_atoms

    @property
    def elements(self):
        if self._elements is None:
            self._elements = []
            for atom in self.mol.GetAtoms():
                self._elements.append(atom.GetAtomicNum())
        return self._elements

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = self.mol.GetConformer().GetPositions()
        return self._coordinates

    @property
    def normal_termination(self):
        if self._normal_termination is None:
            if self.log_path is None:
                self._normal_termination = False
                return self._normal_termination

            end_list = ["Normal", "termination", "of"] if self.gauss_version == 16 else ["Job", "finishes", "at:"]
            self._normal_termination = False
            for line in self.log_lines[-10:]:
                if line.split()[0:3] == end_list:
                    self._normal_termination = True
        return self._normal_termination

    @property
    def charges_mulliken(self):
        """ Get Mulliken charges """
        if self._charges_mulliken is None:
            for i, line in enumerate(self.log_lines_rev):
                if line.startswith(" Mulliken charges:"):
                    charges = []
                    shift = -2
                    while not self.log_lines_rev[i + shift].startswith(" Sum of Mulliken charges"):
                        this_line = self.log_lines_rev[i + shift]
                        charges.append(float(this_line.split()[-1]))
                        shift -= 1
                    self._charges_mulliken = charges
                    break
        return self._charges_mulliken

    @property
    def dipole(self):
        """ Calculate dipole using coordinates and charge for each atom """
        if self._dipole is None:
            if self.charges_mulliken is None:
                return None
            coordinates = self.coordinates
            dipole = [[coordinates[i, 0] * self.charges_mulliken[i], coordinates[i, 1] * self.charges_mulliken[i],
                       coordinates[i, 2] * self.charges_mulliken[i]] for i in range(self.n_atoms)]
            dipole = np.sum(dipole, axis=0)
            self._dipole = dipole
        return self._dipole

    def get_data_frame(self) -> pd.DataFrame:
        """
        :return: a single line dataframe in eV unit
        """
        try:
            pd_dict = {key: [self.prop_dict[key]] for key in self.prop_dict}
            return pd.DataFrame(pd_dict).set_index("sample_id")
        except Exception as e:
            if not self.supress_warning:
                print(f"ERROR while process dataframe {self.base_name}: {e}")
            return pd.DataFrame({"sample_id": self.base_name}, index=[0]).set_index("sample_id")

    def get_error_lines(self) -> pd.DataFrame:
        lines_track = 10
        error_lines = {"f_name": osp.basename(self.log_path)}
        for i in range(1, lines_track + 1):
            error_lines[f"error_line_-{i}"] = [self.log_lines[-i]]
        return pd.DataFrame(error_lines)

    def get_torch_data(self, add_edge=False):
        from torch_geometric.data import Data

        try:
            _tmp_data = self.get_basic_dict()
            prop_dict_pt = copy.deepcopy(self.prop_dict)
            self.conv_type(prop_dict_pt)
            _tmp_data.update(prop_dict_pt)

            this_data = Data(**_tmp_data)
            if add_edge:
                from geometry_processors.DataPrepareUtils import my_pre_transform
                this_data = my_pre_transform(this_data, edge_version="cutoff", do_sort_edge=True, cal_efg=False,
                                             cutoff=10.0, boundary_factor=100., use_center=True, mol=None,
                                             cal_3body_term=False,
                                             bond_atom_sep=False, record_long_range=True)
        except Exception as e:
            print(f"ERROR while process torch data {self.base_name}: {e}")
            this_data = None
        return this_data

    def get_basic_dict(self):
        return {"R": torch.as_tensor(self.coordinates).view(-1, 3),
                "Z": torch.as_tensor(self.elements).view(-1),
                "N": torch.as_tensor(self.n_atoms).view(-1)}

    @staticmethod
    def add_item(info_dict, key, result_dict):
        data = info_dict[key]
        if isinstance(data, str):
            result_dict[key] = data
        elif isinstance(data, int):
            result_dict[key] = torch.as_tensor(data).long()
        elif isinstance(data, float):
            result_dict[key] = torch.as_tensor(data).double()
        elif isinstance(data, torch.Tensor):
            result_dict[key] = data

    @staticmethod
    def conv_type(info_dict):
        for key in info_dict:
            Gauss16Log.add_item(info_dict, key, info_dict)


class Gauss16LogListeners:
    @staticmethod
    def get_all_listeners():
        ins = Gauss16LogListeners()
        return [ins.rotation_consts, ins.dipole_moments, ins.iso_polar, ins.alpha_eig, ins.r2, ins.zpve, ins.U0, ins.U,
                ins.H, ins.G, ins.Cv, ins.E, ins.wall_time]

    @staticmethod
    def rotation_consts(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Rotational constants'):
            vals = line.split()
            out_dict['A'] = float(vals[-3])
            out_dict['B'] = float(vals[-2])
            out_dict['C'] = float(vals[-1])
            return True
        return False

    @staticmethod
    def dipole_moments(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Dipole moment'):
            out_dict['mu'] = float(lines[i - 1].split()[-1])
            return True
        return False

    @staticmethod
    def iso_polar(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Isotropic polarizability'):
            out_dict['alpha'] = float(line.split()[-2])
            return True
        return False

    @staticmethod
    def alpha_eig(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Alpha  occ. eigenvalues') and lines[i - 1].startswith(' Alpha virt. eigenvalues'):
            out_dict['ehomo(eV)'] = float(lines[i - 1].split()[4]) * kwargs["hartree2ev"]
            out_dict['elumo(eV)'] = float(line.split()[-1]) * kwargs["hartree2ev"]
            out_dict['egap(eV)'] = out_dict['ehomo(eV)'] - out_dict['elumo(eV)']
            return True
        return False

    @staticmethod
    def r2(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Electronic spatial extent'):
            out_dict['R2'] = float(line.split()[-1])
            return True
        return False

    @staticmethod
    def zpve(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Zero-point correction'):
            out_dict['zpve(eV)'] = float(line.split()[-2]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def U0(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Sum of electronic and zero-point Energies'):
            out_dict['U0(eV)'] = float(line.split()[-1]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def U(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Sum of electronic and thermal Energies'):
            out_dict['U(eV)'] = float(line.split()[-1]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def H(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Sum of electronic and thermal Enthalpies'):
            out_dict['H(eV)'] = float(line.split()[-1]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def G(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Sum of electronic and thermal Free Energies'):
            out_dict['G(eV)'] = float(line.split()[-1]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def Cv(i, line, lines, out_dict, **kwargs):
        if line.startswith(' Total       '):
            out_dict['Cv'] = float(line.split()[-2])
            return True
        return False

    @staticmethod
    def E(i, line, lines, out_dict, **kwargs):
        if line.startswith(' SCF Done'):
            out_dict['E(eV)'] = float(line.split()[4]) * kwargs["hartree2ev"]
            return True
        return False

    @staticmethod
    def wall_time(i, line, lines, out_dict, **kwargs):
        if line.startswith(" Elapsed time:       "):
            split = line.split()
            days = float(split[2])
            hours = float(split[4])
            minutes = float(split[6])
            seconds = float(split[8])
            total_time = seconds + 60 * (minutes + 60 * (hours + 24 * days))
            out_dict["wall_time(secs)"] = total_time
            return True
        return False


if __name__ == '__main__':
    read_log(software="gauss")
