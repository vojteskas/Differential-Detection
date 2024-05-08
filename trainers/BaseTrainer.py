import torch

from .utils import calculate_EER

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

    def calculate_EER(self, labels, predictions, plot_det: bool, det_subtitle: str) -> float:
        return calculate_EER(type(self.model).__name__, labels, predictions, plot_det, det_subtitle)
