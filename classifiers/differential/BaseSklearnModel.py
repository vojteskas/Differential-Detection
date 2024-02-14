import torch

class BaseSklearnModel:
    """
    Base class for all Sklearn models
    """
    def __init__(self, extractor, feature_processor):
        """
        Base init for defining the extractor and feature_processor
        """
        self.extractor = extractor
        self.feature_processor = feature_processor

    def __call__(self, input_data_ground_truth, input_data_tested):
        """
        Predict classes and probabilities of the tested data. See predict() method.

        It is a wrapper for the predict method for consistency with other (PyTorch) classifiers.
        """
        return self.predict(input_data_ground_truth, input_data_tested)

    def fit(self, bonafide_features, spoof_features):
        raise NotImplementedError("Child classes of BaseSklearnModel need to implement fit() method!")
        
    def predict(self, input_data_ground_truth, input_data_tested):
        raise NotImplementedError("Child classes of BaseSklearnModel need to implement predict() method!")



class SklearnSaver:
    """
    Class to save the Sklearn model to a file
    Needs to be custom because the non-PyTorch model can contain a PyTorch component (extractor, feature_processor)
    PyTorch models can only be saved as a state_dicts, which is exactly what this class does - extract the state_dicts
    and save them along with the rest of the model

    param model: LDAGaussianDiff model to save
    """

    def __init__(self, model: BaseSklearnModel):
        self.model = model

        if isinstance(self.model.extractor, torch.nn.Module):
            self.extractor_state_dict = self.model.extractor.state_dict()
            self.model.extractor = None
        if isinstance(self.model.feature_processor, torch.nn.Module):
            self.feature_processor_state_dict = self.model.feature_processor.state_dict()
            self.model.feature_processor = None
