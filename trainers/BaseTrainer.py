import torch
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_curve
import numpy as np


class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

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

    def train(self):
        raise NotImplementedError("Child classes need to implement train method")

    def val(self):
        raise NotImplementedError("Child classes need to implement val method")

    def eval(self):
        raise NotImplementedError("Child classes need to implement eval method")

    def save_model(self, path: str):
        """
        Save the model to the given path
        If model is a PyTorch model, it will be saved using torch.save(state_dict)
        Problem is when non-PyTorch model contains a Pytorch component (e.g. extractor). In that case,
        the trainer should implement custom saving/loading methods.

        param path: Path to save the model to
        """
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            raise NotImplementedError(
                "Child classes for non-PyTorch models need to implement save_model method"
            )

    def load_model(self, path: str):
        """
        Load the model from the given path
        Try to load the model as a PyTorch model using torch.load,
        otherwise, the child class trainer should implement custom loading method.

        param path: Path to load the model from
        """
        try:
            self.model.load_state_dict(torch.load(path))
        except FileNotFoundError:
            raise
        except:  # Path correct, but not a PyTorch model
            raise NotImplementedError(
                "Child classes for non-PyTorch models need to implement load_model method"
            )

    def calculate_EER(self, labels, predictions) -> float:
        """
        Calculate the Equal Error Rate (EER) from the labels and predictions
        """
        fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=0)
        fnr = 1 - tpr

        # eer from fpr and fnr can differ a bit (its a approximation), so we compute both and take the average
        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = (eer_1 + eer_2) / 2
        return eer
