#!/usr/bin/env python3

from sys import argv

from torch.utils.data import DataLoader

from config import local_config, metacentrum_config
from common import CLASSIFIERS, DATASETS, EXTRACTORS, TRAINERS
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor
from parse_arguments import parse_args

# classifiers
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.SVMDiff import SVMDiff

# trainers
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer

from datasets.utils import custom_pair_batch_create, custom_single_batch_create


def main():
    config = metacentrum_config if "--metacentrum" in argv else local_config

    args = parse_args()

    extractor = EXTRACTORS[args.extractor]()  # map the argument to the class and instantiate it
    processor = MHFAProcessor() if args.processor == "MHFA" else MeanProcessor()

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
        case _:
            try:
                model = CLASSIFIERS[str(args.classifier)][0](extractor, processor, in_dim=extractor.feature_size)
                trainer = TRAINERS[str(args.classifier)](model)
            except KeyError:
                raise ValueError(
                    f"Invalid classifier, should be one of: {list(CLASSIFIERS.keys())}"
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
