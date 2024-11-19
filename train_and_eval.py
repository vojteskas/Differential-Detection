#!/usr/bin/env python3
from common import build_model, get_dataloaders
from config import local_config, metacentrum_config, sge_config
from parse_arguments import parse_args

# trainers
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.BaseSklearnTrainer import BaseSklearnTrainer


def main():
    args = parse_args()

    config = sge_config if args.sge else metacentrum_config if args.metacentrum else local_config

    model, trainer = build_model(args)

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
    )

    # TODO: Implement training of MHFA and AASIST with SkLearn models

    print(f"Training on {type(train_dataloader.dataset).__name__} dataloader.")

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 20
        trainer.train(train_dataloader, val_dataloader, numepochs=args.num_epochs)
        trainer.eval(eval_dataloader, subtitle=str(args.num_epochs))  # Eval after training

    elif isinstance(trainer, BaseSklearnTrainer):
        # Default value of variant = all
        trainer.train(train_dataloader, val_dataloader, variant=args.variant)
        trainer.eval(eval_dataloader)  # Eval after training

    else:
        # Should not happen, should inherit from BaseSklearnTrainer or BaseFFTrainer
        raise ValueError("Invalid trainer, should inherit from BaseSklearnTrainer or BaseFFTrainer.")


if __name__ == "__main__":
    main()
