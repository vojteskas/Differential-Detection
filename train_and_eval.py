#!/usr/bin/env python
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sys import argv

from classifiers.differential.FFDiff import FFDiff
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from datasets.ASVspoof2019 import ASVspoof2019Dataset, custom_batch_create
from embeddings.XLSR.XLSR_300M import XLSR_300M
from feature_processors.MeanProcessor import MeanProcessor
from trainers.FFDiffTrainer import FFDiffTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer

from config import local_config, metacentrum_config


if __name__ == "__main__":
    if "--metacentrum" in argv:
        config = metacentrum_config
    elif "--local" in argv:
        config = local_config
    else:
        raise Exception(
            "You need to specify the configuration.\nUse --metacentrum for running on metacentrum or --local for running locally."
        )

    # Load the dataset
    train_dataset = ASVspoof2019Dataset(
        root_dir=config["data_dir"], protocol_file_name=config["train_protocol"], variant="train"
    )
    val_dataset = ASVspoof2019Dataset(
        root_dir=config["data_dir"], protocol_file_name=config["dev_protocol"], variant="dev"
    )
    eval_dataset = ASVspoof2019Dataset(
        root_dir=config["data_dir"], protocol_file_name=config["eval_protocol"], variant="eval"
    )

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]
    weighted_sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

    # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=custom_batch_create,
        sampler=weighted_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], collate_fn=custom_batch_create, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config["batch_size"], collate_fn=custom_batch_create, shuffle=True
    )

    model = LDAGaussianDiff(XLSR_300M(), MeanProcessor(dim=(0, 2)))
    trainer = LDAGaussianDiffTrainer(model)
    trainer.train(train_dataloader, val_dataloader)
    # trainer.eval(eval_dataloader)
