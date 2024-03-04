#!/usr/bin/env python3

from sys import argv

from torch.utils.data import DataLoader

from config import local_config, metacentrum_config
from common import DATASETS, EXTRACTORS
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor
from parse_arguments import parse_args

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
from trainers.FFConcatTrainer import FFConcatTrainer

from datasets.utils import custom_pair_batch_create, custom_single_batch_create


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
        case "FFConcat4":
            model = FFConcat4(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
        case "FFConcat5":
            model = FFConcat5(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
            config["batch_size"] //= 2  # Half the batch size for FFConcat5
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
                "Only FF, FFConcat{1,2,3,4,5}, FFDiff, GMMDiff, LDAGaussianDiff and SVMDiff classifiers are currently supported."
            )

    print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    dataset = args.dataset
    eval_dataset_class = DATASETS[dataset]
    dataset_config = {}
    if "ASVspoof2019LA" in dataset:
        dataset_config = config["asvspoof2019la"]
    elif "ASVspoof2021" in dataset:
        dataset_config = config["asvspoof2021la"] if "LA" in dataset else config["asvspoof2021df"]
    elif "InTheWild" in dataset:
        dataset_config = config["inthewild"]
    else:
        raise ValueError("Invalid dataset name.")

    if "2021DF" in args.dataset:  # locally there is only a part of 2021DF dataset
        eval_dataset = eval_dataset_class(
            root_dir=config["data_dir"] + dataset_config["eval_subdir"],
            protocol_file_name=dataset_config["eval_protocol"],
            variant="eval",
            local=True if "--local" in config["argv"] else False,
        )
    else:
        eval_dataset = eval_dataset_class(
            root_dir=config["data_dir"] + dataset_config["eval_subdir"],
            protocol_file_name=dataset_config["eval_protocol"],
            variant="eval",
        )
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], collate_fn=collate_func, shuffle=True
    )

    print(
        f"Evaluating {args.checkpoint} {type(model).__name__} {type(eval_dataloader.dataset).__name__} dataloader."
    )

    trainer.eval(eval_dataloader)


if __name__ == "__main__":
    main()
