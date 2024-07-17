import torch
from tqdm import tqdm
import numpy as np

from classifiers.single_input.FF import FF
from trainers.BaseFFTrainer import BaseFFTrainer


class FFTrainer(BaseFFTrainer):
    def __init__(self, model: FF, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        losses = []
        accuracies = []
        # Training loop
        for _, wf, label in tqdm(train_dataloader):

            wf = wf.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, probs = self.model(wf)

            # Loss and backpropagation
            loss = self.lossfn(logits, label.long())
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            predicted = torch.argmax(probs, 1)
            correct = (predicted == label).sum().item()
            accuracy = correct / len(label)

            losses.append(loss.item())
            accuracies.append(accuracy)

        return accuracies, losses

    def val_epoch(
        self, val_dataloader, save_scores=False
    ) -> tuple[list[float], list[float], list[float], list[int], list[str]]:
        losses = []
        labels = []
        scores = []
        predictions = []
        file_names = []

        for file_name, wf, label in tqdm(val_dataloader):
            wf = wf.to(self.device)
            label = label.to(self.device)

            logits, probs = self.model(wf)
            if any(torch.isnan(label)):
                loss = np.inf
            else:
                loss = self.lossfn(logits, label.long()).item()

            predictions.extend(torch.argmax(probs, 1).tolist())

            if save_scores:
                file_names.extend(file_name)
            losses.append(loss)
            labels.extend(label.tolist())
            scores.extend(probs[:, 0].tolist())

        return losses, labels, scores, predictions, file_names
