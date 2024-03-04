#!/usr/bin/env python3
from sys import argv

from config import local_config, metacentrum_config
from common import EXTRACTORS, get_dataloaders
from parse_arguments import parse_args

# feature_processors
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor

# classifiers
from classifiers.differential.FFDiff import FFDiff
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.single_input.FF import FF
from classifiers.differential.FFConcat import FFConcat1, FFConcat2, FFConcat3, FFConcat4, FFConcat5

# trainers
from trainers.FFDiffTrainer import FFDiffTrainer
from trainers.FFTrainer import FFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer
from trainers.BaseSklearnTrainer import BaseSklearnTrainer
from trainers.FFConcatTrainer import FFConcatTrainer


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
        case "FF":
            model = FF(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFTrainer(model)
        case "FFConcat1":
            model = FFConcat1(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat2":
            model = FFConcat2(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat3":
            # Concatenating the features from the two audio files results in twice the feature input size
            model = FFConcat3(extractor, processor, in_dim=extractor.feature_size * 2)
            trainer = FFConcatTrainer(model)
        case "FFConcat4":
            # Using transformer instead of concatenation
            model = FFConcat4(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat5":
            # Does not actually use the processor, but the processor is still required - #TODO: fix this
            model = FFConcat5(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
            config["batch_size"] //= 2  # Half the batch size for FFConcat5
        case "FFDiff":
            model = FFDiff(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFDiffTrainer(model)
        case "GMMDiff":
            gmm_params = {  # Dict comprehension, get gmm parameters from args and remove None values
                k: v for k, v in args.items() if (k in ["n_components", "covariance_type"] and k is not None)
            }
            model = GMMDiff(extractor, processor, **gmm_params if gmm_params else {})  # pass as kwargs
            trainer = GMMDiffTrainer(model)
        case "LDAGaussianDiff":
            model = LDAGaussianDiff(extractor, processor)
            trainer = LDAGaussianDiffTrainer(model)
        case "SVMDiff":
            model = SVMDiff(extractor, processor, kernel=args.kernel if args.kernel else "rbf")
            trainer = SVMDiffTrainer(model)
        case _:
            raise ValueError(
                "Only FF, FFConcat{1,2,3,4,5}, FFDiff, GMMDiff, LDAGaussianDiff and SVMDiff classifiers are currently supported."
            )

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(dataset=args.dataset, config=config)

    # TODO: Implement training of MHFA with SkLearn models
    # skipping for now, focusing on FF(Diff)

    print(
        f"Training {type(model).__name__} model with {type(extractor).__name__} extractor and {type(processor).__name__} processor on {type(train_dataloader.dataset).__name__} dataloader."
    )

    # Train the model
    if isinstance(trainer, (FFDiffTrainer, FFTrainer, FFConcatTrainer)):
        # Default value of numepochs = 100
        trainer.train(train_dataloader, val_dataloader, numepochs=args.num_epochs)
        trainer.eval(eval_dataloader)  # Eval after training

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
        # Should not happen, either FFDiffTrainer or should inherit from BaseSklearnTrainer
        raise ValueError(
            "Invalid trainer, should be either FF(Diff)Trainer or should inherit from BaseSklearnTrainer."
        )


if __name__ == "__main__":
    main()
