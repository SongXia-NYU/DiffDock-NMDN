from subprocess import run
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="command you want to run")
    args = parser.parse_args()

    cmd = args.c
    run("module purge", shell=True)
    run("singularity exec --overlay ~/conda_envs/pytorch1.9-cuda102-15GB-500K.ext3:ro "
        "--overlay ~/conda_envs/pytorch1.9-cuda102-xtb-mamba-2GB-1.3M.ext3:ro "
        "--overlay ~/conda_envs/pytorch1.9-cuda102-snakemake-10GB-400K.ext3:ro "
        "/scratch/work/public/singularity/cuda10.2-cudnn8-devel-ubuntu18.04.sif "
        f"bash -c ' {cmd} '")


if __name__ == '__main__':
    main()
