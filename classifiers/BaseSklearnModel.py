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
