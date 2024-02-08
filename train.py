#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader

from classifiers.differential.FFDiff import FFDiff
from datasets.ASVspoof2019 import ASVspoof2019Dataset, custom_batch_create
from embeddings.XLSR.XLSR_300M import XLSR_300M
from trainers.FFDiffTrainer import FFDiffTrainer

from config import local_config


if __name__ == "__main__":
    # Load the dataset
    train_dataset = ASVspoof2019Dataset(
        root_dir=local_config["data_dir"], protocol_file_name=local_config["train_protocol"], variant="train"
    )

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(train_dataset))

    # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
    dataloader = DataLoader(
        train_dataset, batch_size=local_config["batch_size"], collate_fn=custom_batch_create, sampler=sampler
    )

    model = FFDiff(XLSR_300M())
    trainer = FFDiffTrainer(model)

    trainer.train(dataloader, numepochs=local_config["num_epochs"])
