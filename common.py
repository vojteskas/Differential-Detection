# region Imports
from argparse import Namespace
from typing import Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Classifiers
from classifiers.BaseSklearnModel import BaseSklearnModel
from classifiers.differential.FFConcat import FFLSTM, FFLSTM2, FFConcat1, FFConcat2, FFConcat3
from classifiers.differential.FFDiff import FFDiff, FFDiffAbs, FFDiffQuadratic
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.FFBase import FFBase
from classifiers.single_input.FF import FF

# Config
from config import local_config, metacentrum_config, sge_config

# Datasets
from datasets.ASVspoof5 import ASVspoof5Dataset_pair, ASVspoof5Dataset_single
from datasets.ASVspoof2019 import ASVspoof2019LADataset_pair, ASVspoof2019LADataset_single
from datasets.ASVspoof2021 import (
    ASVspoof2021DFDataset_pair,
    ASVspoof2021DFDataset_single,
    ASVspoof2021LADataset_pair,
    ASVspoof2021LADataset_single,
)
from datasets.InTheWild import InTheWildDataset_pair, InTheWildDataset_single
from datasets.Morphing import MorphingDataset_pair, MorphingDataset_single
from datasets.MLDF import mldf_dataset, mldf_dataloader
from datasets.utils import custom_pair_batch_create, custom_single_batch_create

# Extractors
from extractors.HuBERT import HuBERT_base, HuBERT_extralarge, HuBERT_large
from extractors.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from extractors.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from extractors.XLSR import XLSR_1B, XLSR_2B, XLSR_300M

# Feature processors
from feature_processors.AASIST import AASIST
from feature_processors.MeanProcessor import MeanProcessor
from feature_processors.MHFA import MHFA
from feature_processors.SLS import SLS

# Trainers
from trainers.BaseTrainer import BaseTrainer
from trainers.FFPairTrainer import FFPairTrainer
from trainers.FFTrainer import FFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer

# endregion

# map of argument names to the classes
EXTRACTORS: dict[str, type] = {
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
    "FF": (FF, {}),
    "FFConcat1": (FFConcat1, {}),
    "FFConcat2": (FFConcat2, {}),
    "FFConcat3": (FFConcat3, {}),
    "FFDiff": (FFDiff, {}),
    "FFDiffAbs": (FFDiffAbs, {}),
    "FFDiffQuadratic": (FFDiffQuadratic, {}),
    "FFLSTM": (FFLSTM, {}),
    "FFLSTM2": (FFLSTM2, {}),
    "GMMDiff": (GMMDiff, {"n_components": int, "covariance_type": str}),
    "LDAGaussianDiff": (LDAGaussianDiff, {}),
    "SVMDiff": (SVMDiff, {"kernel": str}),
}
TRAINERS = {  # Maps the classifier to the trainer
    "FF": FFTrainer,
    "FFConcat1": FFPairTrainer,
    "FFConcat2": FFPairTrainer,
    "FFConcat3": FFPairTrainer,
    "FFDiff": FFPairTrainer,
    "FFDiffAbs": FFPairTrainer,
    "FFDiffQuadratic": FFPairTrainer,
    "FFLSTM": FFPairTrainer,
    "FFLSTM2": FFPairTrainer,
    "GMMDiff": GMMDiffTrainer,
    "LDAGaussianDiff": LDAGaussianDiffTrainer,
    "SVMDiff": SVMDiffTrainer,
}
DATASETS = {  # map the dataset name to the dataset class
    "ASVspoof2019LADataset_single": ASVspoof2019LADataset_single,
    "ASVspoof2019LADataset_pair": ASVspoof2019LADataset_pair,
    "ASVspoof2021LADataset_single": ASVspoof2021LADataset_single,
    "ASVspoof2021LADataset_pair": ASVspoof2021LADataset_pair,
    "ASVspoof2021DFDataset_single": ASVspoof2021DFDataset_single,
    "ASVspoof2021DFDataset_pair": ASVspoof2021DFDataset_pair,
    "InTheWildDataset_single": InTheWildDataset_single,
    "InTheWildDataset_pair": InTheWildDataset_pair,
    "MorphingDataset_single": MorphingDataset_single,
    "MorphingDataset_pair": MorphingDataset_pair,
    "ASVspoof5Dataset_single": ASVspoof5Dataset_single,
    "ASVspoof5Dataset_pair": ASVspoof5Dataset_pair,
}


def get_dataloaders(
    dataset: str = "ASVspoof2019LADataset_pair",
    config: dict = metacentrum_config,
    lstm: bool = False,
    augment: bool = False,
    eval_only: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader] | DataLoader:

    # Get the dataset class and config
    # Always train on ASVspoof2019LA, evaluate on the specified dataset (except ASVspoof5)
    dataset_config = {}
    t = "pair" if "pair" in dataset else "single"
    if "ASVspoof2019LA" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2019la"]
    elif "ASVspoof2021" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2021la"] if "LA" in dataset else config["asvspoof2021df"]
    elif "InTheWild" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["inthewild"]
    elif "Morphing" in dataset:
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["morphing"]
    elif "ASVspoof5" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof5"]
    elif "MLDF" in dataset:
        return mldf_dataloader
    else:
        raise ValueError("Invalid dataset name.")

    # Common parameters
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    bs = config["batch_size"] if not lstm else config["lstm_batch_size"]  # Adjust batch size for LSTM models

    # Load the datasets
    train_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    val_dataloader = DataLoader(Dataset())  # dummy dataloader for type hinting compliance
    if not eval_only:
        print("Loading training datasets...")
        train_dataset = train_dataset_class(
            root_dir=config["data_dir"] + dataset_config["train_subdir"],
            protocol_file_name=dataset_config["train_protocol"],
            variant="train",
            augment=augment,
            rir_root=config["rir_root"],
        )

        dev_kwargs = {  # kwargs for the dataset class
            "root_dir": config["data_dir"] + dataset_config["dev_subdir"],
            "protocol_file_name": dataset_config["dev_protocol"],
            "variant": "dev",
        }
        if "2021DF" in dataset:  # 2021DF has a local variant
            dev_kwargs["local"] = True if "--local" in config["argv"] else False
            dev_kwargs["variant"] = "progress"
            val_dataset = eval_dataset_class(**dev_kwargs)
        else:
            # Create the dataset based on dynamically created dev_kwargs
            val_dataset = train_dataset_class(**dev_kwargs)

        # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
        # samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]  # old and slow solution
        samples_weights = np.vectorize(train_dataset.get_class_weights().__getitem__)(
            train_dataset.get_labels()
        )  # blazing fast solution
        weighted_sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

        # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=bs,
            collate_fn=collate_func,
            sampler=weighted_sampler,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=bs, collate_fn=collate_func, shuffle=True)

    print("Loading eval dataset...")
    eval_kwargs = {  # kwargs for the dataset class
        "root_dir": config["data_dir"] + dataset_config["eval_subdir"],
        "protocol_file_name": dataset_config["eval_protocol"],
        "variant": "eval",
    }
    if "2021DF" in dataset:  # 2021DF has a local variant
        eval_kwargs["local"] = True if "--local" in config["argv"] else False

    # Create the dataset based on dynamically created eval_kwargs
    eval_dataset = eval_dataset_class(**eval_kwargs)
    eval_dataloader = DataLoader(eval_dataset, batch_size=bs, collate_fn=collate_func, shuffle=True)

    if eval_only:
        return eval_dataloader
    else:
        return train_dataloader, val_dataloader, eval_dataloader


def build_model(args: Namespace) -> Tuple[FFBase | BaseSklearnModel, BaseTrainer]:
    # Beware of MHFA or AASIST with SkLearn models, they are not implemented yet
    if args.processor in ["MHFA", "AASIST", "SLS"] and args.classifier in ["GMMDiff", "SVMDiff", "LDAGaussianDiff"]:
        raise NotImplementedError("Training of SkLearn models with MHFA, AASIST or SLS is not yet implemented.")
    # region Extractor
    extractor = EXTRACTORS[args.extractor]()  # map the argument to the class and instantiate it
    # endregion

    # region Processor (pooling)
    processor = None
    if args.processor == "MHFA":
        input_transformer_nb = extractor.transformer_layers
        input_dim = extractor.feature_size

        processor_output_dim = (
            input_dim  # Output the same dimension as input - might want to play around with this
        )
        compression_dim = processor_output_dim // 8
        head_nb = round(
            input_transformer_nb * 4 / 3
        )  # Half random guess number, half based on the paper and testing

        processor = MHFA(
            head_nb=head_nb,
            input_transformer_nb=input_transformer_nb,
            inputs_dim=input_dim,
            compression_dim=compression_dim,
            outputs_dim=processor_output_dim,
        )
    elif args.processor == "AASIST":
        processor = AASIST(
            inputs_dim=extractor.feature_size,
            # compression_dim=extractor.feature_size // 8,  # compression dim is hardcoded at the moment
            outputs_dim=extractor.feature_size,  # Output the same dimension as input, might want to play around with this
        )
    elif args.processor == "SLS":
        processor = SLS(
            inputs_dim=extractor.feature_size,
            outputs_dim=extractor.feature_size,  # Output the same dimension as input, might want to play around with this
        )
    elif args.processor == "Mean":
        processor = MeanProcessor()  # default avg pooling along the transformer layers and time frames
    else:
        raise ValueError("Only AASIST, MHFA, Mean and SLS processors are currently supported.")
    # endregion

    # region Model and trainer
    model: FFBase | BaseSklearnModel
    trainer = None
    match args.classifier:
        # region Special case Sklearn models
        case "GMMDiff":
            gmm_params = {  # Dict comprehension, get gmm parameters from args and remove None values
                k: v for k, v in args.items() if (k in ["n_components", "covariance_type"] and k is not None)
            }
            model = GMMDiff(extractor, processor, **gmm_params if gmm_params else {})  # pass as kwargs
            trainer = GMMDiffTrainer(model)
        case "SVMDiff":
            model = SVMDiff(extractor, processor, kernel=args.kernel if args.kernel else "rbf")
            trainer = SVMDiffTrainer(model)
        case "LDAGaussianDiff":
            model = LDAGaussianDiff(extractor, processor)
            trainer = LDAGaussianDiffTrainer(model)
        # endregion
        case _:
            try:
                model = CLASSIFIERS[str(args.classifier)][0](
                    extractor, processor, in_dim=extractor.feature_size
                )
                trainer = TRAINERS[str(args.classifier)](model)
            except KeyError:
                raise ValueError(f"Invalid classifier, should be one of: {list(CLASSIFIERS.keys())}")
    # endregion

    # Print model info
    print(f"Building {type(model).__name__} model with {type(model.extractor).__name__} extractor", end="")
    if isinstance(model, FFBase):
        print(f" and {type(model.feature_processor).__name__} processor.")
    else:
        print(".")

    return model, trainer
