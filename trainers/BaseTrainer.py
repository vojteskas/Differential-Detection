import torch
from scipy.stats import norm
from sklearn.metrics import det_curve, DetCurveDisplay
import numpy as np
import matplotlib.pyplot as plt


class BaseTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

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

    def calculate_EER(self, labels, predictions, plot_det) -> float:
        """
        Calculate the Equal Error Rate (EER) from the labels and predictions
        """
        fpr, fnr, _ = det_curve(labels, predictions, pos_label=0)

        # eer from fpr and fnr can differ a bit (its an approximation), so we compute both and take the average
        eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = (eer_fpr + eer_fnr) / 2

        # Display the DET curve
        if plot_det:
            # eer_fpr_probit = norm.ppf(eer_fpr)
            # eer_fnr_probit = norm.ppf(eer_fnr)
            eer_probit = norm.ppf(eer)

            DetCurveDisplay(fpr=fpr, fnr=fnr, pos_label=0).plot()
            # plt.plot(
            #     eer_fpr_probit,
            #     eer_fpr_probit,
            #     marker="o",
            #     markersize=5,
            #     label=f"EER from FPR: {eer:.2f}",
            #     color="blue",
            # )
            # plt.plot(
            #     eer_fnr_probit,
            #     eer_fnr_probit,
            #     marker="o",
            #     markersize=5,
            #     label=f"EER from FNR: {eer:.2f}",
            #     color="green",
            # )
            plt.plot(eer_probit, eer_probit, marker="o", markersize=4, label=f"EER: {eer:.2f}", color="red")
            plt.legend()
            plt.title(f"DET Curve {type(self.model).__name__}")
            plt.savefig(f"./{type(self.model).__name__}_DET.png")

        return eer
