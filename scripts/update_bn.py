import argparse
from utils.eval.trained_folder import TrainedFolder

def update_bn():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_folder")
    args = parser.parse_args()
    args = vars(args)
    updater = TrainedFolder(args["trained_folder"])
    updater.update_bn()

if __name__ == "__main__":
    update_bn()
