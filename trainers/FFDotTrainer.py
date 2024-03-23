from matplotlib.pylab import f
import numpy as np
import torch
from torch.nn import BCELoss
import matplotlib.pyplot as plt
from tqdm import tqdm

from classifiers.differential.FFDot import FFDot
from trainers.BaseTrainer import BaseTrainer


class FFDotTrainer(BaseTrainer):
    def __init__(self, model: FFDot, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model)

        self.lossfn = BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters()
        )  # Can play with lr and weight_decay for regularization
        self.device = device

        self.model = model.to(device)

        # A statistics tracker dict for the training and validation losses, accuracies and EERs
        self.statistics = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "val_eers": [],
        }

    def train(self, train_dataloader, val_dataloader, numepochs=20, start_epoch=1):
        """
        Train the model on the given dataloader for the given number of epochs
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to train for
        param start_epoch: Epoch to start from (1-indexed)
        """
        for epoch in range(start_epoch, start_epoch + numepochs):  # 1-indexed epochs
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            self.model.train()  # Set model to training mode

            # For accuracy computation in the epoch
            losses = []
            accuracies = []

            # Training loop
            for gt, test, label in tqdm(train_dataloader):

                gt = gt.to(self.device)
                test = test.to(self.device)
                label = label.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                prob = self.model(gt, test)

                # Compute loss
                loss = self.lossfn(prob, label.long())
                loss.backward()
                self.optimizer.step()

                # Compute accuracy
                acc = (prob.round() == label).float().mean()

                losses.append(loss.item())
                accuracies.append(acc.item())

            # Save epoch statistics
            epoch_accuracy = np.mean(accuracies)
            epoch_loss = np.mean(losses)
            print(
                f"Epoch {epoch} finished,", 
                f"training loss: {epoch_loss},", 
                f"training accuracy: {epoch_accuracy}"
            )

            self.statistics["train_losses"].append(epoch_loss)
            self.statistics["train_accuracies"].append(epoch_accuracy)

            self.save_model(f"./FFDot_{epoch}.pt")

            # Validation
            val_loss, val_accuracy, eer = self.val(val_dataloader)
            print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy}")
            print(f"Validation EER: {eer*100}%")
            self.statistics["val_losses"].append(val_loss)
            self.statistics["val_accuracies"].append(val_accuracy)
            self.statistics["val_eers"].append(eer)

    def val(self, val_dataloader):
        """
        Validate the model on the given dataloader and return the loss, accuracy and EER

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(loss, accuracy, EER)
        """

        self.model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            losses = []
            # For EER computation
            labels = []
            scores = []
            predictions = []

            for gt, test, label in tqdm(val_dataloader):
                gt = gt.to(self.device)
                test = test.to(self.device)
                label = label.to(self.device)

                prob = self.model(gt, test)
                loss = self.lossfn(prob, label.long())

                predictions.extend(prob.round().tolist())
                losses.append(loss.item())
                labels.extend(label.tolist())
                scores.extend(prob.tolist())

            val_loss = np.mean(losses).astype(float)
            val_accuracy = np.mean(np.array(labels) == np.array(predictions))
            eer = self.calculate_EER(labels, scores)

            return val_loss, val_accuracy, eer

    def eval(self, eval_dataloader):
        """
        Evaluate the model on the given dataloader and print the loss, accuracy and EER

        param eval_dataloader: Dataloader loading the test data
        """

        # Reuse code from val() to evaluate the model on the eval set
        eval_loss, eval_accuracy, eer = self.val(eval_dataloader)
        print(f"Eval loss: {eval_loss}, eval accuracy: {eval_accuracy}")
        print(f"Eval EER: {eer*100}%")
