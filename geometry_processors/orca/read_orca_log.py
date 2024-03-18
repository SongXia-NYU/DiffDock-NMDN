from geometry_processors.gauss import read_gauss_log


class OrcaLog(read_gauss_log.Gauss16Log):
    def __init__(self, log_path, log_sdf):
        super().__init__(log_path, log_sdf)

    @property
    def normal_termination(self):
        if self._normal_termination is None:
            end_list = ["****ORCA", "TERMINATED", "NORMALLY****"]
            self._normal_termination = False
            for line in self.log_lines[-10:]:
                if line.split()[0:3] == end_list:
                    self._normal_termination = True
        return self._normal_termination

    def get_listeners(self):
        return OrcaLogListeners.get_all_listeners()

    @property
    def charges_mulliken(self):
        """ Get Mulliken charges """
        if self._charges_mulliken is None:
            for i, line in enumerate(self.log_lines_rev):
                if line.startswith("MULLIKEN ATOMIC CHARGES"):
                    charges = []
                    shift = -2
                    while not self.log_lines_rev[i + shift].startswith("Sum of atomic charges:"):
                        this_line = self.log_lines_rev[i + shift]
                        charges.append(float(this_line.split()[-1]))
                        shift -= 1
        return self._charges_mulliken


class OrcaLogListeners:
    @staticmethod
    def get_all_listeners():
        ins = OrcaLogListeners()
        return [ins.total_energy, ins.wall_time]

    @staticmethod
    def total_energy(i, line, lines, out_dict, **kwargs):
        if line.startswith('FINAL SINGLE POINT ENERGY'):
            out_dict['total_energy(Eh)'] = float(line.split()[-1])
            return True
        return False

    @staticmethod
    def wall_time(i, line, lines, out_dict, **kwargs):
        if line.startswith("TOTAL RUN TIME: "):
            split = line.split()
            days = float(split[3])
            hours = float(split[5])
            minutes = float(split[7])
            seconds = float(split[9])
            msec = float(split[11])
            total_time = seconds + 60 * (minutes + 60 * (hours + 24 * days)) + 0.001 * msec
            out_dict["wall_time(secs)"] = total_time
            return True
        return False


if __name__ == '__main__':
    read_gauss_log.read_log(software="orca")
