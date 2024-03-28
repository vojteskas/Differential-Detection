#!/usr/bin/env python3
from sys import argv

from classifiers.differential.FFDot import FFDot
from config import local_config, metacentrum_config
from common import CLASSIFIERS, EXTRACTORS, TRAINERS, get_dataloaders
from parse_arguments import parse_args

# feature_processors
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor

# classifiers
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.SVMDiff import SVMDiff

# trainers
from trainers.BaseFFPairTrainer import BaseFFPairTrainer
from trainers.BaseFFTrainer import BaseFFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer
from trainers.BaseSklearnTrainer import BaseSklearnTrainer


def main():
    config = metacentrum_config if "--metacentrum" in argv else local_config

    args = parse_args()

    extractor = EXTRACTORS[args.extractor]()  # map the argument to the class and instantiate it

    processor = None
    if args.processor == "MHFA":
        input_transformer_nb = extractor.transformer_layers
        input_dim = extractor.feature_size

        processor_output_dim = (
            input_dim  # Output the same dimension as input - might want to play around with this
        )
        compression_dim = processor_output_dim // 8
        head_nb = round(
            input_transformer_nb * 4 / 3
        )  # Half random guess number, half based on the paper and testing

        processor = MHFAProcessor(
            head_nb=head_nb,
            input_transformer_nb=input_transformer_nb,
            inputs_dim=input_dim,
            compression_dim=compression_dim,
            outputs_dim=processor_output_dim,
        )
    elif args.processor == "Mean":
        processor = MeanProcessor()  # default avg pooling along the transformer layers and time frames
    else:
        raise ValueError("Only MHFA and Mean processors are currently supported.")

    model = None
    trainer = None
    match args.classifier:
        case "GMMDiff":
            gmm_params = {  # Dict comprehension, get gmm parameters from args and remove None values
                k: v for k, v in args.items() if (k in ["n_components", "covariance_type"] and k is not None)
            }
            model = GMMDiff(extractor, processor, **gmm_params if gmm_params else {})  # pass as kwargs
            trainer = GMMDiffTrainer(model)
        case "SVMDiff":
            model = SVMDiff(extractor, processor, kernel=args.kernel if args.kernel else "rbf")
            trainer = SVMDiffTrainer(model)
        case "FFDot":
            model = FFDot(extractor, processor)
            trainer = BaseFFPairTrainer(model)
        case _:
            try:
                model = CLASSIFIERS[str(args.classifier)][0](
                    extractor, processor, in_dim=extractor.feature_size
                )
                trainer = TRAINERS[str(args.classifier)](model)
            except KeyError:
                raise ValueError(f"Invalid classifier, should be one of: {list(CLASSIFIERS.keys())}")

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset, config=config, lstm=True if "LSTM" in args.classifier else False
    )

    # TODO: Implement training of MHFA with SkLearn models
    # skipping for now, focusing on FF(Diff)

    print(
        f"Training {type(model).__name__} model with {type(extractor).__name__} extractor and {type(processor).__name__} processor on {type(train_dataloader.dataset).__name__} dataloader."
    )

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 20
        trainer.train(train_dataloader, val_dataloader, numepochs=args.num_epochs)
        trainer.eval(eval_dataloader, subtitle=str(args.num_epochs))  # Eval after training

    elif isinstance(trainer, BaseSklearnTrainer):
        if isinstance(processor, MHFAProcessor):
            # SkLearn models with MHFAProcessor not yet implemented
            raise NotImplementedError(
                "Training of SkLearn models with MHFAProcessor is not yet implemented."
            )
        # Default value of variant = all
        trainer.train(train_dataloader, val_dataloader, variant=args.variant)
        trainer.eval(eval_dataloader)  # Eval after training

    else:
        # Should not happen, should inherit from BaseSklearnTrainer or BaseFFTrainer
        raise ValueError("Invalid trainer, should inherit from BaseSklearnTrainer or BaseFFTrainer.")


if __name__ == "__main__":
    main()
