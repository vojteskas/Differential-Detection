import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from classifiers.differential.FFDiff import FFDiff
from trainers.BaseTrainer import BaseTrainer


class FFDiffTrainer(BaseTrainer):
    def __init__(self, model: FFDiff, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model)

        self.lossfn = CrossEntropyLoss()  # Should also try with BCELoss
        self.optimizer = torch.optim.Adam(model.parameters())  # Can play with lr and weight_decay for regularization
        self.device = device

        model = model.to(device)

    def train(self, train_dataloader, numepochs=100):
        """
        Train the model on the given dataloader for the given number of epochs
        Uses the optimizer and loss function defined in the constructor

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param numepochs: Number of epochs to train for
        """
        model = self.model
        optimizer = self.optimizer
        lossfn = self.lossfn

        epoch_accuracies = []
        epoch_losses = []

        model.train()

        for epoch in range(1, numepochs + 1):  # 1-indexed epochs
            print(f"Starting epoch {epoch} with {len(train_dataloader)} batches")

            # For accuracy computation in the epoch
            losses = []
            accuracies = []

            for i, (gt, test, label) in enumerate(train_dataloader):
                print(f"Batch {i} of {len(train_dataloader)}")
                gt = gt.to(self.device)
                test = test.to(self.device)
                label = label.to(self.device)

                # Check if dataloader balances the classes correctly
                # bonafide += torch.count_nonzero(label).item()
                # spoof += len(label) - torch.count_nonzero(label).item()
                # print(f"Sampled {label}, count so far: {bonafide} bonafide, {spoof} spoof")

                # Forward pass
                optimizer.zero_grad()
                logits, probs = model(gt, test)

                # Loss and backpropagation
                loss = lossfn(logits, label.long())
                loss.backward()
                optimizer.step()

                # Compute accuracy
                predicted = torch.argmax(probs, 1)
                correct = (predicted == label).sum().item()
                accuracy = correct / len(label)

                losses.append(loss.item())
                accuracies.append(accuracy)

                break

            # Save epoch statistics
            epoch_accuracy = np.mean(accuracies)
            epoch_loss = np.mean(losses)
            print(f"Epoch {epoch} finished, loss: {np.mean(losses)}, accuracy: {np.mean(accuracies)}")

            epoch_accuracies.append(epoch_accuracy)
            epoch_losses.append(epoch_loss)

            # Every 10 epochs
            # TODO: validate the model on the validation set
            # plot losses and accuracy and save the model
            if epoch % 10 == 0:
                self._plot_loss_accuracy(epoch_losses, epoch_accuracies, f"Epoch {epoch}")
                self.save_model(f"./FFDiff_{epoch}.pt")

            # TODO: Enable early stopping based on validation accuracy

    def val(self):
        pass

    def eval(self):
        pass

    def _plot_loss_accuracy(self, losses, accuracies, subtitle: str = ""):
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss")
        plt.plot(accuracies, label="Accuracy")
        plt.legend()
        plt.title("FFDiff Loss and Accuracy" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.savefig(f"./FFDiff_loss_acc_{subtitle}.png")
