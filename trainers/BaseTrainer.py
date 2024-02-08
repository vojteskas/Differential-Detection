import torch
import joblib


class BaseTrainer:
    def __init__(self, model):
        self.model = model

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

        param path: Path to save the model to
        """
        if isinstance(self.model, torch.nn.Module):
            torch.save(self.model.state_dict(), path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        """
        Load the model from the given path
        Try to load the model as a PyTorch model first using torch.load
        If that fails, try to load it as an sklearn model using joblib.load

        param path: Path to load the model from
        """
        try:
            self.model.load_state_dict(torch.load(path))
        except:
            self.model = joblib.load(path)
