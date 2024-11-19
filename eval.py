#!/usr/bin/env python3
from torch.utils.data import DataLoader

from config import local_config, metacentrum_config, sge_config
from common import build_model, get_dataloaders
from parse_arguments import parse_args


def main():
    args = parse_args()

    config = sge_config if args.sge else metacentrum_config if args.metacentrum else local_config

    model, trainer = build_model(args)

    print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    # Load the dataset
    eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        eval_only=True,
    )
    assert isinstance( # Is here for type checking and hinting compliance
        eval_dataloader, DataLoader
    ), "Error type of eval_dataloader returned from get_dataloaders."

    print(
        f"Evaluating {args.checkpoint} {type(model).__name__} on "
        + f"{type(eval_dataloader.dataset).__name__} dataloader."
    )

    trainer.eval(eval_dataloader, subtitle=str(args.checkpoint))


if __name__ == "__main__":
    main()
