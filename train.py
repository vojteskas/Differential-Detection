#!/usr/bin/env python3

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from diff_model import DiffModel
from datasets import ASVspoof2019Dataset, custom_batch_create
import matplotlib.pyplot as plt
from sys import argv

# TODO: Add adaptive learning rate (Adam already has it?)

def train(model: DiffModel, dataloader: DataLoader, device: torch.device):
    print(f"Using device: {device}, starting training...")

    lossfn = CrossEntropyLoss(weight=dataloader.dataset.get_class_weights())
    optimizer = torch.optim.Adam(model.parameters())

    model = model.to(device)
    model.train()

    losses = []
    accuracies = []

    bonafide, spoof = 0, 0

    for i, (gt, test, label) in enumerate(dataloader):
        gt = gt.transpose(1, 2).to(device) # transpose to the shape (batch_size, embedding_size, seq_len = 1)
        test = test.transpose(1, 2).to(device) # transpose to the shape (batch_size, embedding_size, seq_len = 1)
        label = label.to(device)

        bonafide += np.count_nonzero(label)
        spoof += len(label) - np.count_nonzero(label)
        print(f"Sampled {label}, count so far: {bonafide} bonafide, {spoof} spoof")

        optimizer.zero_grad()
        vals, probs = model(gt, test)

        loss = lossfn(vals, label.long())
        loss.backward()
        optimizer.step()

        # compute accuracy
        predicted = torch.argmax(probs, 1)
        correct = (predicted == label).sum().item()
        accuracy = correct / len(label)

        losses.append(loss.item())
        accuracies.append(accuracy)

        if i % 10 == 0:
            print(f"[{i}/{len(dataloader)}] Loss: {loss.item()}")
        if "--local" in argv and i == 10:
            break

    # save the model
    torch.save(model.state_dict(), "./diffmodel.pt")

    # plot loss and accuracy
    print("Training finished, plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label="Loss")
    plt.plot(accuracies, label="Accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("TrainingLossAndAccuracy.png")


if __name__ == "__main__":
    # TODO: Better path handling
    data_dir = "/mnt/e/VUT/Deepfakes/Datasets/LA" if "--local" in argv else "./LA"

    # d = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Needs about 224GB RAM
    d = torch.device("cpu") # Use CPU for now
    model = DiffModel(device=d)
    dataset = ASVspoof2019Dataset(
        root_dir=data_dir,
        protocol_file_name="ASVspoof2019.LA.cm.train.trn.txt",
        variant="train",
    )
    samples_weights = [dataset.get_class_weights()[i] for i in dataset.get_labels()]
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_batch_create, sampler=sampler)
    train(model, dataloader, d)
