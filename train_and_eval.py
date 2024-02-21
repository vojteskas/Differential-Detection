#!/usr/bin/env python3
from typing import Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
from sys import argv

from config import local_config, metacentrum_config
from parse_arguments import parse_args, EXTRACTORS

# datasets
from datasets.ASVspoof2019 import (
    ASVspoof2019LADataset_pair,
    ASVspoof2019LADataset_single,
    custom_pair_batch_create,
    custom_single_batch_create,
)

# feature_processors
from feature_processors.MHFAProcessor import MHFAProcessor
from feature_processors.MeanProcessor import MeanProcessor

# classifiers
from classifiers.differential.FFDiff import FFDiff
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.single_input.FF import FF
from classifiers.differential.FFConcat import FFConcat1, FFConcat2, FFConcat3
from trainers.BaseSklearnTrainer import BaseSklearnTrainer
from trainers.FFConcatTrainer import FFConcatTrainer

# trainers
from trainers.FFDiffTrainer import FFDiffTrainer
from trainers.FFTrainer import FFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer


def get_dataloaders(
    dataset="ASVspoof2019LADataset_pair", config=metacentrum_config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if "ASVspoof2019LA" not in dataset:
        raise NotImplementedError("Only ASVspoof2019LA pair and single datasets are currently supported.")

    dataset_class = ASVspoof2019LADataset_single if "single" in dataset else ASVspoof2019LADataset_pair
    # Load the dataset
    train_dataset = dataset_class(
        root_dir=config["data_dir"], protocol_file_name=config["train_protocol"], variant="train"
    )
    val_dataset = dataset_class(
        root_dir=config["data_dir"], protocol_file_name=config["dev_protocol"], variant="dev"
    )
    eval_dataset = dataset_class(
        root_dir=config["data_dir"], protocol_file_name=config["eval_protocol"], variant="eval"
    )

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]
    weighted_sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

    # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collate_func,
        sampler=weighted_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], collate_fn=collate_func, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], collate_fn=collate_func, shuffle=True
    )

    return train_dataloader, val_dataloader, eval_dataloader


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
            model = FFConcat3(extractor, processor, in_dim=extractor.feature_size)
            trainer = FFConcatTrainer(model)
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
                "Only FFDiff, GMMDiff, LDAGaussianDiff and SVMDiff classifiers are currently supported."
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
