import numpy as np
import torch
from tqdm import tqdm
from typing import Literal

from classifiers.differential.GMMDiff import GMMDiff
from trainers.BaseTrainer import BaseTrainer

class GMMDiffTrainer(BaseTrainer):
    def __init__(self, model: GMMDiff, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model)

        self.device = device  # Much faster to use GPU at least for the feature extraction

        self.model.extractor = self.model.extractor.to(device)

    # Looks like there is no partial_fit method for GMM
    # so we need to extract all the features and then fit the model
    # Apart from fitting all the data, there are alternative options:
    # TODO: 1. (Avg) pool the batch of features and use that to fit the model
    # TODO: 2. Random sample the dataset for a 'reasonable number' of samples to fit the model
    # TODO: 3. Write 'partial_fit' manually
    def train(
        self, train_dataloader, val_dataloader, variant: Literal["all", "avg_pool", "random_sample"] = "all"
    ):
        """
        Train the model on the given dataloader for the given number of epochs

        param train_dataloader: Dataloader loading the training data
        """

        # TODO: Consider doing PCA/LDA on the features before fitting the GMMs

        if variant == "all":
            self._train_all(train_dataloader)
        else:
            raise NotImplementedError(f"Training variant {variant} not implemented")

        self.save_model(f"./GMMDiff_{variant}.pt")

        print("GMMDiff model trained")
        print("Validating model")
        validation_accuracy, validation_eer = self.val(val_dataloader)
        print(f"Validation accuracy: {validation_accuracy}, validation EER: {validation_eer}")

    def _train_all(self, train_dataloader):
        """
        Train the model on all the data in the given dataloader
        """
        print("Training GMMDiff model on all data")

        extractor = self.model.extractor
        feature_processor = self.model.feature_processor
        bonafide_features = []
        spoof_features = []

        for gt, test, label in tqdm(train_dataloader):
            feats_gt = extractor.extract_features(gt.to(self.device))
            feats_test = extractor.extract_features(test.to(self.device))

            feats_gt = feature_processor(feats_gt)
            feats_test = feature_processor(feats_test)

            diff = feats_gt - feats_test

            bonafide_features.extend(diff[label == 1].tolist())
            spoof_features.extend(diff[label == 0].tolist())

        self.model.bonafide_classifier.fit(bonafide_features)
        self.model.spoof_classifier.fit(spoof_features)

    def val(self, val_dataloader) -> tuple[float, float]:
        """
        Validate the model on the given dataloader

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(accuracy, EER)
        """
        labels = []
        scores = []  # Posterior probabilities
        class_predictions = []  # MAP predictions - defined in the classifier

        for gt, test, label in tqdm(val_dataloader):
            class_predictions, probs = self.model(gt, test)

            class_predictions.extend(class_predictions.tolist())
            scores.extend(probs.tolist())
            labels.extend(label.tolist())

        val_accuracy = np.mean(np.array(labels) == np.array(class_predictions))
        eer = self.calculate_EER(labels, scores)

        return val_accuracy, eer

    def eval(self, eval_dataloader):
        """
        Evaluate the model on the given dataloader

        param eval_dataloader: Dataloader loading the evaluation/test data

        return: Tuple(accuracy, EER)
        """
        print("Evaluating GMMDiff model")
        # Reuse code from val() to evaluate the model on the eval set
        eval_accuracy, eer = self.val(eval_dataloader)
        print(f"Eval accuracy: {eval_accuracy}")
        print(f"Eval EER: {eer}")
