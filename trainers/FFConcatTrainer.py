import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from tqdm import tqdm

from trainers.BaseTrainer import BaseTrainer


class FFConcatTrainer(BaseTrainer):
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model)

        # Mabye TODO??? Add class weights for the loss function - maybe not necessary since we have weighted sampler
        self.lossfn = CrossEntropyLoss()  # Should also try with BCELoss
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

    def train(self, train_dataloader, val_dataloader, numepochs=100, start_epoch=1):
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

            # Save epoch statistics
            epoch_accuracy = np.mean(accuracies)
            epoch_loss = np.mean(losses)
            print(
                f"Epoch {epoch} finished,", 
                f"training loss: {np.mean(losses)},", 
                f"training accuracy: {np.mean(accuracies)}"
            )

            self.statistics["train_losses"].append(epoch_loss)
            self.statistics["train_accuracies"].append(epoch_accuracy)

            # Every epoch
            # plot losses and accuracy and save the model
            # validate on the validation set (incl. computing EER)
            self._plot_loss_accuracy(
                self.statistics["train_losses"],
                self.statistics["train_accuracies"],
                f"Training epoch {epoch}",
            )
            self.save_model(f"./{type(self.model).__name__}_{epoch}.pt")

            # Validation
            val_loss, val_accuracy, eer = self.val(val_dataloader)
            print(f"Validation loss: {val_loss}, validation accuracy: {val_accuracy}")
            print(f"Validation EER: {eer*100}%")
            self.statistics["val_losses"].append(val_loss)
            self.statistics["val_accuracies"].append(val_accuracy)
            self.statistics["val_eers"].append(eer)

            # TODO: Enable early stopping based on validation accuracy/loss/EER

        self._plot_loss_accuracy(
            self.statistics["val_losses"], self.statistics["val_accuracies"], "Validation"
        )
        self._plot_eer(self.statistics["val_eers"], "Validation")

    def val(self, val_dataloader) -> tuple[float, float, float]:
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
                # print(f"Validation batch {i+1} of {len(val_dataloader)}")

                gt = gt.to(self.device)
                test = test.to(self.device)
                label = label.to(self.device)

                logits, probs = self.model(gt, test)
                loss = self.lossfn(logits, label.long())

                predictions.extend(torch.argmax(probs, 1).tolist())

                losses.append(loss.item())
                labels.extend(label.tolist())
                scores.extend(probs[:, 0].tolist())

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

    def _plot_loss_accuracy(self, losses, accuracies, subtitle: str = ""):
        """
        Plot the loss and accuracy and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(losses, label="Loss")
        plt.plot(accuracies, label="Accuracy")
        plt.legend()
        plt.title(f"{type(self.model).__name__} Loss and Accuracy" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.savefig(f"./{type(self.model).__name__}_loss_acc_{subtitle}.png")

    def _plot_eer(self, eers, subtitle: str = ""):
        """
        Plot the EER and save the graph to a file
        """
        plt.figure(figsize=(12, 6))
        plt.plot(eers, label="EER")
        plt.legend()
        plt.title(f"{type(self.model).__name__} EER" + f" - {subtitle}" if subtitle else "")
        plt.xlabel("Epoch")
        plt.ylabel("EER")
        plt.savefig(f"./{type(self.model).__name__}_EER_{subtitle}.png")
