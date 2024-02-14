import torch
import joblib
from sklearn.metrics import roc_curve
import numpy as np


class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

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
        If model is not a PyTorch model (i.e. from sklearn), it will be saved using joblib.dump
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
        Try to load the model as a PyTorch model first using torch.load
        If that fails, try to load it as an sklearn model using joblib.load
        Problem is when non-PyTorch model contains a Pytorch component (e.g. extractor). In that case,
        the trainer should implement custom saving/loading methods.

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
