import torch

from BaseTrainer import BaseTrainer
from classifiers.differential.BaseSklearnModel import BaseSklearnModel

class BaseSklearnTrainer(BaseTrainer):
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

    def save_model(self, path: str):
        """
        Save the Sklearn model to the given path
        Problem is when the model contains a Pytorch component (e.g. extractor). In that case,
        the PyTorch components need to be extracted as state_dicts and saved separately.

        param path: Path to save the model to
        """

class SklearnSaver:
    """
    Class to save the Sklearn model to a file
    Needs to be custom because the non-PyTorch model can contain a PyTorch component (extractor, feature_processor)
    PyTorch models can only be saved as a state_dicts, which is exactly what this class does - extract the state_dicts
    and save them along with the rest of the model

    param model: Sklearn model to save
    """

    def __init__(self, model: BaseSklearnModel):
        self.model = model

        if isinstance(self.model.extractor, torch.nn.Module):
            self.extractor_state_dict = self.model.extractor.state_dict()
            self.model.extractor = None
        if isinstance(self.model.feature_processor, torch.nn.Module):
            self.feature_processor_state_dict = self.model.feature_processor.state_dict()
            self.model.feature_processor = None
