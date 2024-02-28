#!/usr/bin/env python3

from sys import argv

from matplotlib.pylab import f

from config import local_config, metacentrum_config
from common import EXTRACTORS, get_dataloaders
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor
from parse_arguments import parse_args

# classifiers
from classifiers.differential.FFDiff import FFDiff
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.single_input.FF import FF
from classifiers.differential.FFConcat import FFConcat1, FFConcat2, FFConcat3

# trainers
from trainers.FFDiffTrainer import FFDiffTrainer
from trainers.FFTrainer import FFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer
from trainers.FFConcatTrainer import FFConcatTrainer


def main():
    config = metacentrum_config if "--metacentrum" in argv else local_config

    args = parse_args()

    extractor = EXTRACTORS[args.extractor]()  # map the argument to the class and instantiate it
    processor = MHFAProcessor() if args.processor == "MHFA" else MeanProcessor()

    match args.classifier:
        case "FF":
            model = FF(extractor, processor, in_dim=extractor.feature_size)  # FF model does not use processor
            trainer = FFTrainer(model)
        case "FFConcat1":
            model = FFConcat1(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat2":
            model = FFConcat2(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat3":
            model = FFConcat3(extractor, processor, in_dim=extractor.feature_size * 2)
            trainer = FFConcatTrainer(model)
        case "FFDiff":
            model = FFDiff(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFDiffTrainer(model)
        case "GMMDiff":
            model = GMMDiff(None, None)  # pass as kwargs
            trainer = GMMDiffTrainer(model)
        case "LDAGaussianDiff":
            model = LDAGaussianDiff(None, None)
            trainer = LDAGaussianDiffTrainer(model)
        case "SVMDiff":
            model = SVMDiff(None, None)
            trainer = SVMDiffTrainer(model)
        case _:
            raise ValueError(
                "Only FF, FFConcat{1,2,3}, FFDiff, GMMDiff, LDAGaussianDiff and SVMDiff classifiers are currently supported."
            )

    print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    _, _, eval_dataloader = get_dataloaders(dataset=args.dataset, config=config)

    print(
        f"Evaluating {args.checkpoint} {type(model).__name__} {type(eval_dataloader.dataset).__name__} dataloader."
    )

    trainer.eval(eval_dataloader)


if __name__ == "__main__":
    main()
