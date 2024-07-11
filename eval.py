#!/usr/bin/env python3

from sys import argv

from torch.utils.data import DataLoader, IterableDataset

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
    elif "Morphing" in dataset:
        dataset_config = config["morphing"]
    elif "ASVspoof5" in dataset:
        dataset_config = config["asvspoof5"]
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
    shuffle = False if isinstance(eval_dataset, IterableDataset) else True
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], collate_fn=collate_func, shuffle=shuffle
    )

    print(
        f"Evaluating {args.checkpoint} {type(model).__name__} {type(eval_dataloader.dataset).__name__} dataloader."
    )

    trainer.eval(eval_dataloader, subtitle=str(args.checkpoint))


if __name__ == "__main__":
    main()
