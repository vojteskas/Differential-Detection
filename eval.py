#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from datasets import ASVspoof2019Dataset
from diff_model import DiffModel


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()  # set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # don't compute gradients
        for i, (gt, test, label) in enumerate(dataloader):
            gt = gt.squeeze(0).to(device)
            test = test.squeeze(0).to(device)
            label = label.to(device)

            output = model(gt, test).unsqueeze(0)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if i % 10 == 0:
                print(f"[{i}/{len(dataloader)}]")

            if i >= 100:
                break

    accuracy = correct / total
    print(f"Accuracy of the model on the {total} test images: {accuracy * 100}%")


if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {d}")
    # Load the trained model
    model = DiffModel(device=d)
    model.load_state_dict(torch.load("./diffmodel.pt"))

    # Create a DataLoader for the validation dataset
    validation_dataset = ASVspoof2019Dataset(
        root_dir="/mnt/e/VUT/Deepfakes/Datasets/LA",
        protocol_file_name="ASVspoof2019.LA.cm.dev.trl.txt",
        variant="dev"
    )
    validation_dataloader = DataLoader(validation_dataset, shuffle=True)

    # Evaluate the model
    evaluate(model, validation_dataloader, d)
