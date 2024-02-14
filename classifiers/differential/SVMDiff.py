from matplotlib.pylab import f
import numpy as np
from typing import Literal
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from classifiers.differential.BaseSklearnModel import BaseSklearnModel


class SVMDiff(BaseSklearnModel):
    def __init__(
        self, extractor, feature_processor, kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf"
    ):
        """
        SVM classifier using the difference between two embeddings.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        """

        super().__init__(extractor, feature_processor)

        self.kernel = kernel
        # TODO: SVM are not scale invariant, so we should standardize the data
        # clf = make_pipeline(StandardScaler(), SVC())

        # PARAMETERS TO TUNE:
        # C parameter (regularization), default=1.0, higher values -> less regularization
        # gamma parameter (kernel coefficient), default=1/n_features,
        # # gamma relates to the influence single training example. Larger gamma -> the closer other examples must be to be affected.
        self.classifier = SVC(
            kernel=kernel,
            cache_size=4069,
            degree=7,
        )  # degree=7 for poly kernel, otherwise ignored

    def fit(self, bonafide_features, spoof_features):
        """
        Fit the SVM classifier to the given features.

        param bonafide_features: Features of the bonafide data of shape: (num_samples, num_features)
        param spoof_features: Features of the spoof data of shape: (num_samples, num_features)
        """
        self.classifier.fit(
            np.vstack((bonafide_features, spoof_features)),
            np.hstack((np.zeros(len(bonafide_features)), np.ones(len(spoof_features)))),
        )

    def predict(self, input_data_ground_truth, input_data_tested):
        """
        Predict classes of the tested data.

        param input_data_ground_truth: Ground truth audio data
        param input_data_tested: Audio data to be tested

        return: Tuple(Class predictions, Decision function scores)
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        diff = emb_gt - emb_test
        diff = diff.cpu()  # If computing embeddings on GPU, move to CPU

        class_predictions = self.classifier.predict(diff)

        # decision_function returns the distance of the samples to the separating hyperplane - 1D
        # we need to transform it to 2D array to match the shape for eer calculation
        # Do that by assigning the other class the negative of the score
        scores = self.classifier.decision_function(diff)
        scores = np.vstack((scores, -scores)).T

        return class_predictions, scores  # might also need to return probabilities
