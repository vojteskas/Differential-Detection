#!/usr/bin/env python3

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from diff_model import DiffModel
from datasets import ASVspoof2019Dataset
import matplotlib.pyplot as plt

# TODO: Add adaptive learning rate
# TODO: Update for batch training after adding collate function to dataset


def train(model: DiffModel, dataloader: DataLoader, device: torch.device):
    print(f"Using device: {device}, starting training...")
    lossfn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model = model.to(device)
    model.train()

    losses = []
    accuracies = []

    for i, (gt, test, label) in enumerate(dataloader):
        gt = gt.squeeze(0).to(device)  # Squeeze the first dimension (batch size = 1)
        test = test.squeeze(0).to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(gt, test).unsqueeze(0)

        loss = lossfn(output, label.long())
        loss.backward()
        optimizer.step()

        # compute accuracy
        _, predicted = torch.max(output, 1)
        correct = (predicted == label).sum().item()
        accuracy = correct / len(label)

        losses.append(loss.item())
        accuracies.append(accuracy)

        if i % 10 == 0:
            print(f"[{i}/{len(dataloader)}] Loss: {loss.item()}")

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
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffModel(device=d)
    dataset = ASVspoof2019Dataset(
        root_dir="/mnt/e/VUT/Deepfakes/Datasets/LA",
        protocol_file_name="ASVspoof2019.LA.cm.train.trn.txt",
    )
    dataloader = DataLoader(dataset, shuffle=True)
    train(model, dataloader, d)
