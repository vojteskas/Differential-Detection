import argparse

from common import EXTRACTORS, CLASSIFIERS


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers.")

    # either --metacentrum or --local must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metacentrum", action="store_true", help="Flag for running on metacentrum.")
    group.add_argument("--local", action="store_true", help="Flag for running locally.")

    # Add argument for loading a checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.",
    )

    # dataset
    # TODO: Allow for different datasets
    # TODO: Allow for multiple datasets to be used
    # For now, implicitely use the ASVspoof2019 dataset
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="ASVspoof2019LADataset_pair",
        help="Dataset to be used. One of: ASVspoof2019LADataset_pair, ASVspoof2019LADataset_single",
        required=True,
    )

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
    classifiers = ["FF, FFConcat, FFDiff", "GMMDiff", "LDAGaussianDiff", "SVMDiff"]
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
        default=20,
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
