from utils.train.mdn2pkd_trainer import MDN2PKdTrainer
from utils.train.trainer import Trainer, parse_args


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()