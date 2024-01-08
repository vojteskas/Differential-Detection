#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from datasets import ASVspoof2019Dataset, custom_batch_create
from diff_model import DiffModel
from sys import argv


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()  # set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # don't compute gradients
        for i, (gt, test, label) in enumerate(dataloader):
            gt = gt.transpose(1, 2).to(device)
            test = test.transpose(1, 2).to(device)
            label = label.to(device)

            vals, probs = model(gt, test)
            print(f"output: {probs}")
            predicted = torch.argmax(probs, dim=1)
            print(f"predicted: {predicted}, label: {label}")
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if i % 10 == 0:
                print(f"[{i}/{len(dataloader)}]")

            if "--local" in argv and i == 100:
                break

    accuracy = correct / total
    print(f"Accuracy of the model on the {total} test images: {accuracy * 100}%")

if __name__ == "__main__":
    # TODO: Better path handling
    data_dir = "/mnt/e/VUT/Deepfakes/Datasets/LA" if "--local" in argv else "./LA"

    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {d}")
    # Load the trained model
    model = DiffModel(device=d)
    model.load_state_dict(torch.load("./diffmodel.pt"))

    # Create a DataLoader for the validation dataset
    validation_dataset = ASVspoof2019Dataset(
        root_dir=data_dir,
        protocol_file_name="ASVspoof2019.LA.cm.eval.trl.txt",
        variant="eval"
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, collate_fn=custom_batch_create)

    # Evaluate the model
    evaluate(model, validation_dataloader, d)
