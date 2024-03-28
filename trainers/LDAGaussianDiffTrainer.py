import torch
from typing import Literal

from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from trainers.BaseSklearnTrainer import BaseSklearnTrainer


class LDAGaussianDiffTrainer(BaseSklearnTrainer):
    def __init__(self, model: LDAGaussianDiff, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(model, device)

        # Move the extractor to the device, it's much faster on GPU
        self.model.extractor = self.model.extractor.to(device)

    def train(
        self, train_dataloader, val_dataloader, variant: Literal["all", "avg_pool", "random_sample"] = "all"
    ):
        """
        Train the model on the given dataloader for the given number of epochs

        param train_dataloader: Dataloader loading the training data
        param val_dataloader: Dataloader loading the validation/dev data
        param variant: Variant of sampling the data for training: all, avg_pool, random_sample
        """

        # Variant of sampling the data for training: all, avg_pool, random_sample
        if variant == "all":
            self._train_all(train_dataloader)
        else:
            raise NotImplementedError(f"Training variant {variant} not implemented")

        self.save_model(f"./LDAGaussianDiff_{variant}.pt")

        print(f"LDAGuassianDiff model trained")
        print("Validating model")
        validation_accuracy, validation_eer = self.val(val_dataloader)
        print(f"Validation accuracy: {validation_accuracy}, validation EER: {validation_eer}")

    def val(self, val_dataloader) -> tuple[float, float]:
        """
        Validate the model on the given dataloader

        param val_dataloader: Dataloader loading the validation/dev data

        return: Tuple(accuracy, EER)
        """
        return self._val(val_dataloader)

    def eval(self, eval_dataloader, subtitle: str = ""):
        """
        Evaluate the model on the given dataloader

        param eval_dataloader: Dataloader loading the evaluation/test data
        """
        print("Evaluating LDAGaussianDiff model")
        # Reuse code from val() to evaluate the model on the eval set
        eval_accuracy, eer = self.val(eval_dataloader)
        print(f"Eval accuracy: {eval_accuracy}")
        print(f"Eval EER: {eer}")
