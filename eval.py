#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from datasets import ASVspoof2019Dataset, custom_batch_create
from diff_model import DiffModel
from sys import argv
from w2v_model import SSLModel
from sklearn.metrics import roc_curve
import numpy as np


def evaluate(model, dataloader, device):
    wav2vec = SSLModel(device=device)
    model.to(device)
    model.eval()  # set the model to evaluation mode

    # For Accuracy computation
    total = 0
    correct = 0
    # For EER computation
    labels = []
    predictions = []

    with torch.no_grad():
        for i, (gt, test, label) in enumerate(dataloader):
            gt = gt.transpose(1, 2).to(device)
            test = test.transpose(1, 2).to(device)
            label = label.to(device)

            gt = wav2vec.extract_feat(gt)
            test = wav2vec.extract_feat(test)
            gt = torch.mean(gt, dim=1)
            test = torch.mean(test, dim=1)

            vals, probs = model(gt, test)
            # print(f"output: {probs}")
            predicted = torch.argmax(probs, dim=1)
            labels.extend(label.tolist())
            predictions.extend(probs[:, 0].tolist())
            # print(f"probs: {probs},\npredicted:\n{predicted}\n, label:\n{label}\n")
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if i % 10 == 0:
                print(f"[{i}/{len(dataloader)}: correct {correct} / {total}]")

            if "--local" in argv and i == 100:
                break

    accuracy = correct / total
    print(f"Accuracy of the model on the {total} test images: {accuracy * 100}%")

    # Compute EER
    fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=0)
    fnr = 1 - tpr

    # eer from fpr and fnr can differ a bit (its a approximation), so we compute both and take the average
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2
    print(f"EER: {eer * 100}%")


if __name__ == "__main__":
    # TODO: Better path handling
    data_dir = "/mnt/e/VUT/Deepfakes/Datasets/LA" if "--local" in argv else "./LA"

    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {d}")

    variant = "eval"
    print(f"Using variant: {variant}")
    # Create a DataLoader for the validation dataset
    validation_dataset = ASVspoof2019Dataset(
        root_dir=data_dir, protocol_file_name="ASVspoof2019.LA.cm."+variant+".trl.txt", variant=variant
    )

    samples_weights = [validation_dataset.get_class_weights()[i] for i in validation_dataset.get_labels()]
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(validation_dataset))
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, collate_fn=custom_batch_create, sampler=sampler)

    # validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, collate_fn=custom_batch_create)

    # Evaluate the model
    # numepochs = 1 if "--local" in argv else 11
    # for epoch in range(numepochs):
    model = DiffModel(device=d)
    model.load_state_dict(torch.load(f"./diffmodel_30.pt"))
    # print(f"Loaded model from epoch {epoch}")
    evaluate(model, validation_dataloader, d)
    # print(f"^^^ Evaluated model diffmodel_{epoch}.pt ^^^")
