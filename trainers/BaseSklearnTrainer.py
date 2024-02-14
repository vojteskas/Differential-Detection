from copy import deepcopy
import joblib
import numpy as np
import torch
from tqdm import tqdm

from trainers.BaseTrainer import BaseTrainer
from classifiers.differential.BaseSklearnModel import BaseSklearnModel


class BaseSklearnTrainer(BaseTrainer):
    def __init__(self, model: BaseSklearnModel, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

    def save_model(self, path: str):
        """
        Save the Sklearn model to the given path
        Problem is when the model contains a Pytorch component (e.g. extractor). In that case,
        the PyTorch components need to be extracted as state_dicts and saved separately.

        param path: Path to save the model to
        """
        if isinstance(self.model.extractor, torch.nn.Module) or isinstance(
            self.model.feature_processor, torch.nn.Module
        ):
            serialized_model = SklearnSaver(deepcopy(self.model))
            torch.save(serialized_model, path)
        else:
            joblib.dump(self.model, path)

    def load_model(self, path: str):
        """
        Load the model from the given path

        param path: Path to load the model from
        """
        serialized_model = torch.load(path)
        if isinstance(serialized_model, SklearnSaver):
            self.model = serialized_model.model
            if isinstance(self.model.extractor, torch.nn.Module):
                self.model.extractor.load_state_dict(serialized_model.extractor_state_dict)
            if isinstance(self.model.feature_processor, torch.nn.Module):
                self.model.feature_processor.load_state_dict(serialized_model.feature_processor_state_dict)
        else:
            self.model = joblib.load(path)

    def _train_all(self, train_dataloader):
        """
        Train the model on all the data in the given dataloader
        """
        print(f"Training {self.model.__class__.__name__} model on all data")

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

            bonafide_features.extend(diff[label == 0].tolist())
            spoof_features.extend(diff[label == 1].tolist())

        self.model.fit(np.array(bonafide_features), np.array(spoof_features))

    def _val(self, val_dataloader) -> tuple[float, float]:
        """
        Validate the model on the given dataloader

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(accuracy, EER)
        """
        labels = []
        scores = []
        class_predictions = []

        for gt, test, label in tqdm(val_dataloader):
            batch_predictions, score = self.model(gt.to(self.device), test.to(self.device))

            class_predictions.extend(batch_predictions.tolist())
            scores.extend(score[:, 0].tolist())
            labels.extend(label.tolist())

        # print(f"Labels: {np.array(labels).astype(int)}")
        # print(f"Predic: {np.array(class_predictions)}")
        # print(f"Scores: {np.array(scores)}")

        val_accuracy = np.mean(np.array(labels) == np.array(class_predictions))
        eer = self.calculate_EER(labels, scores)

        return val_accuracy, eer


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
