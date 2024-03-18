import argparse
import os
import os.path as osp
from typing import List

import pandas as pd


def read_xtb(args):
    info_df = pd.DataFrame()
    for log, sdf in zip(args["log_list"], args["sdf_list"]):
        this_info = {}

        sample_id = osp.basename(log).split(".")[0]
        if sdf is not None:
            sample_id_sdf = osp.basename(sdf).split(".")[0]
            assert sample_id == sample_id_sdf, f"inconsistent sample id: {sample_id_sdf}, {sample_id}"
        this_info["sample_id"] = sample_id

        log_info = XTBLog(log, sdf)
        for prop_name in ["total_energy", "homo_lumo_gap", "gradient_norm", "wall_time"]:
            prop, unit = getattr(log_info, prop_name)
            this_info[f"{prop_name}({unit})"] = prop
        for prop_name in ["smiles"]:
            prop = getattr(log_info, prop_name)
            this_info[f"{prop_name}"] = prop

        this_info["log_name"] = osp.basename(log)
        this_info["sdf_name"] = osp.basename(sdf) if sdf is not None else None

        info_df = info_df.append(this_info, ignore_index=True)

    info_df.to_csv(args["out_path"], index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_list", type=str, nargs="*")
    parser.add_argument("--sdf_list", type=str, nargs="*", default=None)
    parser.add_argument("--out_path", type=str)
    args = vars(parser.parse_args())
    if args["sdf_list"] is None:
        args["sdf_list"] = len(args["log_list"]) * [None]
    read_xtb(args)


class XTBLog:
    def __init__(self, f_name, sdf=None):
        self.sdf = sdf
        self.f_name = f_name
        with open(f_name) as f:
            log = f.readlines()
            log.reverse()
            self.log_rev: List[str] = log

        self._wall_time = None
        self._wall_time_unit = None
        self._normal_finish = None
        self._total_energy = None
        self._total_energy_unit = None
        self._gradient_norm = None
        self._gradient_norm_unit = None
        self._homo_lumo_gap = None
        self._homo_lumo_gap_unit = None
        self._mol = None
        self._smiles = None
        if self.sdf is not None:
            from rdkit.Chem import SDMolSupplier
            mols = list(SDMolSupplier(self.sdf))
            assert len(mols) == 1
            mol = mols[0]
            self._mol = mol
            total_energy = mol.GetProp("total energy / Eh")
            gradient_norm = mol.GetProp("gradient norm / Eh/a0")
            assert abs(float(total_energy) - float(self.total_energy[0])) < 1e-9, self.error_msg()
            assert abs(float(gradient_norm) - float(self.gradient_norm[0])) < 1e-9, self.error_msg()

    @property
    def smiles(self):
        if self._smiles is None and self._mol is not None:
            self._smiles = self._mol.GetProp("SMILES")
        return self._smiles

    @property
    def wall_time(self):
        if self._wall_time is None:
            for i, line in enumerate(self.log_rev):
                if line.startswith(" * finished run on"):
                    assert i - 3 >= 0, self.error_msg()
                    info_line = self.log_rev[i-3]
                    assert info_line.startswith(" * wall-time:"), "wall time line: " + info_line
                    info_list = info_line.strip().split()[2:]
                    days = float(info_list[0])
                    hours = float(info_list[2])
                    minutes = float(info_list[4])
                    seconds = float(info_list[6])
                    total_time = seconds + 60 * (minutes + 60 * (hours + 24 * days))
                    self._wall_time = total_time
                    self._wall_time_unit = "secs"
        return self._wall_time, self._wall_time_unit

    @property
    def normal_finish(self):
        if self._normal_finish is None:
            success = False
            for i, line in enumerate(self.log_rev):
                if line.startswith(" * finished run on"):
                    success = True
                    break
            self._normal_finish = success
        return self._normal_finish

    @property
    def total_energy(self):
        if self._total_energy is None:
            for i, line in enumerate(self.log_rev):
                if line.startswith("          | TOTAL ENERGY"):
                    info_list = line.strip().split()
                    self._total_energy = info_list[3]
                    self._total_energy_unit = info_list[4]
        assert self._total_energy_unit is not None
        return self._total_energy, self._total_energy_unit

    @property
    def gradient_norm(self):
        if self._gradient_norm is None:
            for i, line in enumerate(self.log_rev):
                if line.startswith("          | GRADIENT NORM"):
                    info_list = line.strip().split()
                    self._gradient_norm = info_list[3]
                    self._gradient_norm_unit = info_list[4]
        assert self._gradient_norm_unit is not None
        return self._gradient_norm, self._gradient_norm_unit

    @property
    def homo_lumo_gap(self):
        if self._homo_lumo_gap is None:
            for i, line in enumerate(self.log_rev):
                if line.startswith("          | GRADIENT NORM"):
                    info_list = line.strip().split()
                    self._homo_lumo_gap = info_list[3]
                    self._homo_lumo_gap_unit = info_list[4]
        assert self._homo_lumo_gap_unit is not None
        return self._homo_lumo_gap, self._homo_lumo_gap_unit

    def error_msg(self):
        return self.f_name + " has problem :-) Hope you do not see this!"


if __name__ == '__main__':
    main()
