import argparse

from common import CLASSIFIERS


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers.")

    # either --metacentrum, --sge or --local must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metacentrum", action="store_true", help="Flag for running on metacentrum.")
    group.add_argument("--sge", action="store_true", help="Flag for running on SGE on BUT FIT.")
    group.add_argument("--local", action="store_true", help="Flag for running locally.")

    # Add argument for loading a checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.",
    )

    # dataset
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="ASVspoof2019LADataset_pair",
        help="Dataset to be used. See common.DATASETS for available datasets.",
        required=True,
    )

    # extractor
    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        default="XLSR_300M",
        help=f"Extractor to be used. See common.EXTRACTORS for available extractors.",
        required=True,
    )

    # feature processor
    feature_processors = ["MHFA", "AASIST", "Mean"]
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
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        help=f"Classifier to be used. See common.CLASSIFIERS for available classifiers.",
        required=True,
    )

    # augmentations
    parser.add_argument(
        "-a",
        "--augmentations",
        action="store_true",
        help="Flag for whether to use augmentations during training. Does nothing during evaluation.",
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

    # region Optional arguments
    # training
    classifier_args.add_argument(
        "-ep",
        "--num_epochs",
        type=int,
        help="Number of epochs to train for. Does not concern SkLearn classifiers.",
        default=20,
    )

    classifier_args.add_argument(
        "--sampling",
        type=str,
        help="Variant of sampling the data for training SkLearn mocels. One of: all, avg_pool, random_sample.",
        default="all",
    )
    # endregion

    args = parser.parse_args()
    return args
