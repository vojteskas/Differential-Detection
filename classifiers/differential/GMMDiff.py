from typing import Literal
import numpy as np
from sklearn.mixture import GaussianMixture


class GMMDiff:
    def __init__(
        self,
        extractor,
        feature_processor,
        # GMM parameters
        n_components: int = 2,  # Should evaluate and estimate the optimal number https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    ):
        """
        GMM classifier using the difference between two embeddings.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        """

        self.extractor = extractor
        self.feature_processor = feature_processor
        self.n_components = n_components
        self.covariance_type = covariance_type

        self.bonafide_classifier = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type
        )
        self.spoof_classifier = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type
        )  # Maybe different number of components?

    def __call__(self, input_data_ground_truth, input_data_tested):
        """
        Predict classes and probabilities of the tested data. See predict() method.

        It is a wrapper for the predict method for consistency with other (PyTorch) classifiers.
        """
        return self.predict(input_data_ground_truth, input_data_tested)

    def predict(self, input_data_ground_truth, input_data_tested):
        """
        Predict the class of the tested data.

        Extract features from the audio data, process them and predict the class using the class GMMs.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: MAP class predictions and the aposteriori probabilities.
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        diff = emb_gt - emb_test
        diff = diff.cpu()  # If extracting features on GPU, move the result back to cpu

        bonafide_score = self.bonafide_classifier.score_samples(diff)
        spoof_score = self.spoof_classifier.score_samples(diff)

        # Maximum aposteriori (MAP) prediction
        # If spoof prob is higer, get True and map to 1, if bonafide prob is higher, get False and map to 0
        class_predictions = (bonafide_score < spoof_score).astype(int)  # 0 for bonafide, 1 for spoof

        return class_predictions, np.column_stack((bonafide_score, spoof_score))
