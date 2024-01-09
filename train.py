#!/usr/bin/env python3

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from diff_model import DiffModel
from datasets import ASVspoof2019Dataset, custom_batch_create
import matplotlib.pyplot as plt
from sys import argv
from w2v_model import SSLModel

# TODO: Add adaptive learning rate (Adam already has it?)


def train(model: DiffModel, dataloader: DataLoader, device: torch.device):
    print(f"Using device: {device}, starting training...")

    lossfn = CrossEntropyLoss(weight=dataloader.dataset.get_class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters())

    wav2vec = SSLModel(device=device)

    model = model.to(device)
    model.train()

    print(f"Training on {len(dataloader)} batches")

    # For EER computation and plotting
    epoch_acc = []
    epoch_loss = []

    numepochs = 1 if "--local" in argv else 11
    for epoch in range(numepochs):
        print(f"Starting epoch {epoch}")

        # For accuracy computation in the epoch
        losses = []
        accuracies = []

        bonafide, spoof = 0, 0

        for i, (gt, test, label) in enumerate(dataloader):
            gt = gt.to(device).transpose(1, 2)  # transpose to the shape (batch_size, embedding_size, seq_len = 1)
            test = test.to(device).transpose(1, 2)  # transpose to the shape (batch_size, embedding_size, seq_len = 1)
            label = label.to(device)

            # For accuracy computation
            bonafide += torch.count_nonzero(label).item()
            spoof += len(label) - torch.count_nonzero(label).item()
            # print(f"Sampled {label}, count so far: {bonafide} bonafide, {spoof} spoof")

            with torch.no_grad():  # don't compute gradients == no fine-tuning of wav2vec model
                gt = wav2vec.extract_feat(gt)
                test = wav2vec.extract_feat(test)
                gt = torch.mean(gt, dim=1)
                test = torch.mean(test, dim=1)

            optimizer.zero_grad()
            vals, probs = model(gt, test)

            # loss and backpropagation
            loss = lossfn(vals, label.long())
            loss.backward()
            optimizer.step()

            # compute accuracy
            predicted = torch.argmax(probs, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss.item())
            accuracies.append(accuracy)

            # if i % 10 == 0:
            #     print(f"[{i}/{len(dataloader)}] Loss: {loss.item()}")
            if "--local" in argv and i == 100:
                break

        epoch_acc.append(np.mean(accuracies))
        epoch_loss.append(np.mean(losses))
        print(f"Epoch {epoch} finished, loss: {np.mean(losses)}, accuracy: {np.mean(accuracies)}")
        print(f"Sampled {bonafide} bonafide, {spoof} spoof")

        if epoch % 10 == 0:  # Save the model every 10 epochs for comparison and checkpointing
            torch.save(model.state_dict(), f"./diffmodel_{epoch}.pt")

            # plot loss and accuracy
            print(f"Plotting results for epoch {epoch}...")
            plt.figure(figsize=(12, 6))
            plt.plot(losses, label="Loss")
            plt.plot(accuracies, label="Accuracy")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Iteration")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(f"TrainingLossAndAccuracy_{epoch}.png")

    # Save the final model
    torch.save(model.state_dict(), "./diffmodel.pt")
    # plot final loss and accuracy
    print("Training finished, plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_loss, label="Loss")
    plt.plot(epoch_acc, label="Accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("TrainingLossAndAccuracy_Epochs.png")


if __name__ == "__main__":
    # TODO: Better path handling
    data_dir = "/mnt/e/VUT/Deepfakes/Datasets/LA" if "--local" in argv else "./LA"

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Needs about 224GB of memory when finetuning!
    # d = torch.device("cpu") # Use CPU for now

    # model and dataset
    model = DiffModel(device=d)
    dataset = ASVspoof2019Dataset(
        root_dir=data_dir,
        protocol_file_name="ASVspoof2019.LA.cm.train.trn.txt",
        variant="train",
    )

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    samples_weights = [dataset.get_class_weights()[i] for i in dataset.get_labels()]
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(dataset))

    # create dataloader, use custom collate_fn to pad the data to the longes recording in batch
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_batch_create, sampler=sampler)

    # train the model
    train(model, dataloader, d)
