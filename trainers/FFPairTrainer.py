import torch
from tqdm import tqdm

from classifiers.differential.FFConcat import FFConcatBase
from classifiers.differential.FFDiff import FFDiffBase
from trainers.BaseFFTrainer import BaseFFTrainer


class FFPairTrainer(BaseFFTrainer):
    def __init__(self, model: FFDiffBase|FFConcatBase, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

    def train_epoch(self, train_dataloader) -> tuple[list[float], list[float]]:
        """
        Train the model on the given dataloader for one epoch
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        return: Tuple(lists of accuracies, list of losses)
        """
        # For accuracy computation in the epoch
        losses = []
        accuracies = []

        # Training loop
        for _, gt, test, label in tqdm(train_dataloader):
            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits, probs = self.model(gt, test)

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

        for file_name, gt, test, label in tqdm(val_dataloader):
            # print(f"Validation batch {i+1} of {len(val_dataloader)}")

            gt = gt.to(self.device)
            test = test.to(self.device)
            label = label.to(self.device)

            logits, probs = self.model(gt, test)
            loss = self.lossfn(logits, label.long())

            predictions.extend(torch.argmax(probs, 1).tolist())

            if save_scores:
                file_names.extend(file_name)
            losses.append(loss.item())
            labels.extend(label.tolist())
            scores.extend(probs[:, 0].tolist())

        # for name, label, score, prediction in zip(file_names, labels, scores, predictions):
        #     print(f"File: {name}, Score: {score}, Label: {label}, Prediction: {prediction}")

        return losses, labels, scores, predictions, file_names
