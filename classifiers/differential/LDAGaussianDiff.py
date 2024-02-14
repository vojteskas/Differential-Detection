import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

from classifiers.differential.BaseSklearnModel import BaseSklearnModel


class LDAGaussianDiff(BaseSklearnModel):
    def __init__(self, extractor, feature_processor):
        """
        LDA classifier using the difference between two embeddings.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        """

        self.extractor = extractor
        self.feature_processor = feature_processor

        self.lda = LinearDiscriminantAnalysis()

        self.classifier = GaussianNB()

    def fit(self, bonafide_features, spoof_features, plot=False):
        """
        Fit the LDA and Gaussian classifier to the given features.

        param bonafide_features: Features of the bonafide data of shape: (num_samples, num_features)
        param spoof_features: Features of the spoof data of shape: (num_samples, num_features)
        """
        self.lda.fit(
            np.vstack((bonafide_features, spoof_features)),
            np.hstack((np.zeros(len(bonafide_features)), np.ones(len(spoof_features)))),
        )

        bonafide_features = self.lda.transform(bonafide_features)
        spoof_features = self.lda.transform(spoof_features)

        self.classifier.fit(
            np.vstack((bonafide_features, spoof_features)),
            np.hstack((np.zeros(len(bonafide_features)), np.ones(len(spoof_features)))),
        )

        if plot:
            # Plot the gaussian distributions
            plt.figure()
            plt.hist(bonafide_features, bins=20, alpha=0.5, label="bonafide")
            plt.hist(spoof_features, bins=20, alpha=0.5, label="spoof")
            plt.legend(loc="upper right")
            plt.title("Gaussian distributions of LDA features")
            plt.savefig("lda_gaussian_distributions.png")

    def predict(self, input_data_ground_truth, input_data_tested):
        """
        Predict classes and probabilities of the tested data.

        param input_data_ground_truth: Ground truth audio data
        param input_data_tested: Audio data to be tested

        return: Tuple of MAP class predictions and the aposteriori probabilities
        """
        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        diff = emb_gt - emb_test
        diff = diff.cpu()  # If computing embeddings on GPU, move to CPU

        diff = self.lda.transform(diff)

        probs = self.classifier.predict_proba(diff)
        class_predictions = self.classifier.predict(diff)

        return class_predictions, probs
