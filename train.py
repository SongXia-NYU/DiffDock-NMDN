from utils.train.mdn2pkd_trainer import MDN2PKdTrainer
from utils.train.trainer import Trainer, flex_parse


def main():
    args = flex_parse()
    trainer_cls = Trainer if args["mdn2pkd_model"] is None else MDN2PKdTrainer
    trainer = trainer_cls(args)
    trainer.train()


if __name__ == "__main__":
    main()