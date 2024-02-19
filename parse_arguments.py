import argparse
from typing import Dict, Tuple

# classifiers
from classifiers.differential.FFDiff import FFDiff
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff

# datasets
from datasets.ASVspoof2019 import ASVspoof2019Dataset, custom_batch_create

# extractors
from embeddings.HuBERT import HuBERT_base, HuBERT_large, HuBERT_extralarge
from embeddings.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from embeddings.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from embeddings.XLSR import XLSR_300M, XLSR_1B, XLSR_2B

# trainers
from trainers.FFDiffTrainer import FFDiffTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer

# map of argument names to the classes
EXTRACTORS = {
    "HuBERT_base": HuBERT_base,
    "HuBERT_large": HuBERT_large,
    "HuBERT_extralarge": HuBERT_extralarge,
    "Wav2Vec2_base": Wav2Vec2_base,
    "Wav2Vec2_large": Wav2Vec2_large,
    "Wav2Vec2_LV60k": Wav2Vec2_LV60k,
    "WavLM_base": WavLM_base,
    "WavLM_baseplus": WavLM_baseplus,
    "WavLM_large": WavLM_large,
    "XLSR_300M": XLSR_300M,
    "XLSR_1B": XLSR_1B,
    "XLSR_2B": XLSR_2B,
}
CLASSIFIERS: Dict[str, Tuple[type, Dict[str, type]]] = {
    # Maps the classifier to tuples of the corresponding class and the initializable arguments
    "FFDiff": (FFDiff, {}),
    "GMMDiff": (GMMDiff, {"n_components": int, "covariance_type": str}),
    "LDAGaussianDiff": (LDAGaussianDiff, {}),
    "SVMDiff": (SVMDiff, {"kernel": str}),
}
TRAINERS = {  # Maps the classifier to the trainer
    "FFDiff": FFDiffTrainer,
    "GMMDiff": GMMDiffTrainer,
    "LDAGaussianDiff": LDAGaussianDiffTrainer,
    "SVMDiff": SVMDiffTrainer,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers.")

    # either --metacentrum or --local must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metacentrum", action="store_true", help="Flag for running on metacentrum.")
    group.add_argument("--local", action="store_true", help="Flag for running locally.")

    # dataset
    # TODO: Allow for different datasets
    # TODO: Allow for multiple datasets to be used
    # For now, implicitely use the ASVspoof2019 dataset

    # extractor
    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        default="XLSR_300M",
        help=f"Extractor to be used. One of: {', '.join(EXTRACTORS)}",
        required=True,
    )

    # feature processor
    feature_processors = ["MHFA", "Mean"]
    parser.add_argument(
        "-p",
        "--processor",
        "--pooling",
        type=str,
        help=f"Feature processor to be used. One of: {', '.join(feature_processors)}",
        required=True,
    )
    # TODO: Allow for passing parameters to the feature processor (mainly MHFA)

    # classifier
    classifiers = ["FFDiff", "GMMDiff", "LDAGaussianDiff", "SVMDiff"]
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        help=f"Classifier to be used. One of: {', '.join(classifiers)}",
        required=True,
    )

    # Add arguments specific to each classifier
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    classifier_args = parser.add_argument_group("Classifier-specific arguments")
    for classifier, (classifier_class, args) in CLASSIFIERS.items():
        if args:  # if there are any arguments that can be passed to the classifier
            for arg, arg_type in args.items():
                if arg == "kernel":  # only for SVMDiff, display the possible kernels
                    classifier_args.add_argument(
                        f"--{arg}",
                        type=str,
                        help=f"{arg} for {classifier}. One of: {', '.join(kernels)}",
                    )
                    # TODO: Add parameters for the kernels (e.g. degree for poly, gamma for rbf, etc.)
                else:
                    classifier_args.add_argument(f"--{arg}", type=arg_type, help=f"{arg} for {classifier}")

    # maybe TODO: add flag for enabling/disabling evaluation after training

    # TODO: Allow for training from a checkpoint

    # region Optional arguments
    # training
    classifier_args.add_argument(
        "-ep",
        "--num_epochs",
        type=int,
        help="Number of epochs to train for. Will only be used together with FFDiff classifier or MHFA pooling.",
        default=100,
    )

    classifier_args.add_argument(
        "--sampling",
        type=str,
        help="Variant of sampling the data for training a non-NN model. Will only be used together with GMMDiff, LDAGaussianDiff, and SVMDiff classifiers.",
        default="all",
    )
    # endregion

    args = parser.parse_args()
    return args
