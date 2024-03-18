from glob import glob
import os.path as osp


def generate_test_al(str_id=None, start=None, end=None, auto_dataset=False):
    if str_id is None:
        start = start
        end = end+1 if end > 0 else start + 1
        exp_ids = range(start, end)
    else:
        exp_ids = [str_id]

    for exp_id in exp_ids:
        folder_name = f"../exp{exp_id}_active_ALL_*"
        folders = glob(folder_name)
        if len(folders) > 1:
            answer = input(f"exp{exp_id} has more than one running folder, run all of them? Y | N\n")
            if answer.lower() != "y":
                print("exiting...")
                exit(-1)
        for run_folder in folders:
            if auto_dataset:
                if input("You are using auto dataset for Frag20 MMFF, are you sure? Y | N") != "Y":
                    exit(-1)
                # generate custom config file: replacing data provider
                with open(osp.join(run_folder, f"config-exp{exp_id}.txt")) as f_in:
                    config_lines = f_in.readlines()
                with open(osp.join(run_folder, "config-test.txt"), "w") as f_out:
                    for line in config_lines:
                        if line.startswith("--data_provider="):
                            f_out.write("--data_provider=frag9to20_all[geometry=MMFF]\n")
                        else:
                            f_out.write(line)

            # generate submit job file
            with open("../utils/file_templates/subjob-al-test-amd.pbs") as f_in:
                template = f_in.read()
            with open(f"../subjob-al-test-exp{exp_id}.pbs", "w") as f_out:
                folder_base = osp.basename(run_folder)
                f_out.write(template.format(folder_base, folder_base))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--str_id", type=str, default=None)
    parser.add_argument("--auto_dataset", action="store_true")
    args = parser.parse_args()
    args = vars(args)

    generate_test_al(**args)


if __name__ == '__main__':
    main()
