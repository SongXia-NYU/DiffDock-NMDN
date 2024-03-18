import os
import os.path as osp
from glob import glob


def sdf_to_gauss(i, o, keyword):
    os.system("obabel -isdf {} -ocom -xk {} -O {}".format(i, keyword, o))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--keyword")
    args = vars(parser.parse_args())
    sdf_to_gauss(**args)


if __name__ == '__main__':
    pass

