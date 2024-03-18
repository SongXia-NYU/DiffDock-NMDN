class AmberReader:
    def __init__(self, log_file) -> None:
        self.log_file = log_file

        self._log_lines = None
        self._info_dict = None

    def read_log(self):
        raise NotImplemented

    @property
    def info_dict(self):
        if self._info_dict is None:
            self._info_dict = self.read_log()
        return self._info_dict

    @property
    def log_lines(self):
        if self._log_lines is None:
            with open(self.log_file) as f:
                self._log_lines = f.readlines()
        return self._log_lines


class PBSAReader(AmberReader):
    def __init__(self, log_file) -> None:
        super().__init__(log_file)

    def read_log(self):
        info_dict = {}
        active = False
        for line in self.log_lines:
            if line.strip() == "FINAL RESULTS":
                active = True
            
            if active:
                if "=" in line:
                    key = None
                    parts = line.split("=")
                    for part in parts:
                        if key is None:
                            key = part.strip()
                            continue
                        pair = part.strip().split()
                        if len(pair) == 2:
                            val, newkey = pair
                        else:
                            val = pair[0]
                        info_dict[key] = val
                        key = newkey
            if line.startswith("| Total time "):
                info_dict["Total time"] = line.strip().split()[-1]
        return info_dict


class GBSAReader(AmberReader):
    def __init__(self, log_file) -> None:
        super().__init__(log_file)

    def read_log(self):
        info_dict = {}
        active = False
        for i, line in enumerate(self.log_lines):
            line = self.log_lines[i]
            if not active and line.strip() == "FINAL RESULTS":
                active = True
            if not active:
                continue

            if line.strip().startswith("NSTEP"):
                keys = line.strip().split()[1:]
                vals = self.log_lines[i+1].strip().split()[1:]
                for key, val in zip(keys, vals):
                    if key in ["ENERGY", "RMS", "GMAX"]:
                        val = float(val)
                    info_dict[key] = val
            
            if "=" in line:
                parts = line.split("=")
                key = parts[0].strip()
                for part in parts[1:]:
                    pair = part.strip().split()
                    if len(pair) >= 2:
                        val = pair[0]
                        newkey = " ".join(pair[1:])
                    else:
                        val = pair[0]
                    info_dict[key] = val
                    key = newkey
            if line.startswith("| Total time "):
                info_dict["Total time"] = line.strip().split("| Total time ")[-1].split()[0]
        return info_dict


if __name__ == "__main__":
    reader = PBSAReader("/scratch/sx801/scripts/Mol3DGenerator/scripts/AF-SwissProt/test_run_AF-A0A023IWD9-F1-model_v3/AF-A0A023IWD9-F1-model_v3.out")
    print(reader.info_dict)
