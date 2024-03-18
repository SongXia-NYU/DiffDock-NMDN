import glob
import os.path as osp


def generate_ft_al(seed_folder, config, save_root="."):
    config_template = ""
    with open(config) as f:
        for line in f.readlines():
            if line.startswith("--use_trained_model="):
                config_template += "--use_trained_model={}\n"
            else:
                config_template += line

    for i, f in enumerate(glob.glob(osp.join(seed_folder, "exp*_cycle_-1_*"))):
        config_base = osp.basename(config).split(".")[0]
        with open(osp.join(save_root, f"{config_base}-{i}.txt"), "w") as out:
            out.write(config_template.format(f))


if __name__ == '__main__':
    generate_ft_al("../exp400_active_ALL_2021-11-07_203805", "../config-exp405.txt", "..")
