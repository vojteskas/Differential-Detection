from typing import Dict, Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler

# classifiers
from classifiers.differential.FFDiff import FFDiff, FFDiffAbs, FFDiffQuadratic
from classifiers.differential.FFDot import FFDot
from classifiers.differential.GMMDiff import GMMDiff
from classifiers.differential.LDAGaussianDiff import LDAGaussianDiff
from classifiers.differential.SVMDiff import SVMDiff
from classifiers.single_input.FF import FF
from classifiers.differential.FFConcat import FFConcat1, FFConcat2, FFConcat3, FFLSTM, FFLSTM2

# extractors
from extractors.HuBERT import HuBERT_base, HuBERT_large, HuBERT_extralarge
from extractors.Wav2Vec2 import Wav2Vec2_base, Wav2Vec2_large, Wav2Vec2_LV60k
from extractors.WavLM import WavLM_base, WavLM_baseplus, WavLM_large
from extractors.XLSR import XLSR_300M, XLSR_1B, XLSR_2B

# trainers
from trainers.BaseFFPairTrainer import BaseFFPairTrainer
from trainers.FFTrainer import FFTrainer
from trainers.GMMDiffTrainer import GMMDiffTrainer
from trainers.LDAGaussianDiffTrainer import LDAGaussianDiffTrainer
from trainers.SVMDiffTrainer import SVMDiffTrainer

# datasets
from datasets.utils import custom_pair_batch_create, custom_single_batch_create
from datasets.ASVspoof2019 import ASVspoof2019LADataset_pair, ASVspoof2019LADataset_single
from datasets.ASVspoof2021 import (
    ASVspoof2021LADataset_single,
    ASVspoof2021LADataset_pair,
    ASVspoof2021DFDataset_single,
    ASVspoof2021DFDataset_pair,
    ASVspoof2021DFDataset_VC_single,
    ASVspoof2021DFDataset_VC_pair,
    ASVspoof2021DFDataset_nonVC_single,
    ASVspoof2021DFDataset_nonVC_pair,
)
from datasets.InTheWild import InTheWildDataset_pair, InTheWildDataset_single
from datasets.Morphing import MorphingDataset_single, MorphingDataset_pair

from config import local_config, metacentrum_config

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
    "FFDot": (FFDot, {}),
    "FFLSTM": (FFLSTM, {}),
    "FFLSTM2": (FFLSTM2, {}),
    "GMMDiff": (GMMDiff, {"n_components": int, "covariance_type": str}),
    "LDAGaussianDiff": (LDAGaussianDiff, {}),
    "SVMDiff": (SVMDiff, {"kernel": str}),
}
TRAINERS = {  # Maps the classifier to the trainer
    "FF": FFTrainer,
    "FFConcat1": BaseFFPairTrainer,
    "FFConcat2": BaseFFPairTrainer,
    "FFConcat3": BaseFFPairTrainer,
    "FFDiff": BaseFFPairTrainer,
    "FFDiffAbs": BaseFFPairTrainer,
    "FFDiffQuadratic": BaseFFPairTrainer,
    "FFDot": BaseFFPairTrainer,
    "FFLSTM": BaseFFPairTrainer,
    "FFLSTM2": BaseFFPairTrainer,
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
    "ASVspoof2021DFDataset_VC_single": ASVspoof2021DFDataset_VC_single,
    "ASVspoof2021DFDataset_VC_pair": ASVspoof2021DFDataset_VC_pair,
    "ASVspoof2021DFDataset_nonVC_single": ASVspoof2021DFDataset_nonVC_single,
    "ASVspoof2021DFDataset_nonVC_pair": ASVspoof2021DFDataset_nonVC_pair,
    "InTheWildDataset_single": InTheWildDataset_single,
    "InTheWildDataset_pair": InTheWildDataset_pair,
    "MorphingDataset_single": MorphingDataset_single,
    "MorphingDataset_pair": MorphingDataset_pair,
}


def get_dataloaders(
    dataset="ASVspoof2019LADataset_pair", config=metacentrum_config, lstm=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset_config = {}
    if "ASVspoof2019LA" in dataset:
        train_dataset_class = DATASETS[dataset]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2019la"]
    elif "ASVspoof2021" in dataset:
        t = "pair" if "pair" in dataset else "single"
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["asvspoof2021la"] if "LA" in dataset else config["asvspoof2021df"]
    elif "InTheWild" in dataset:
        t = "pair" if "pair" in dataset else "single"
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["inthewild"]
    elif "Morphing" in dataset:
        t = "pair" if "pair" in dataset else "single"
        train_dataset_class = DATASETS[f"ASVspoof2019LADataset_{t}"]
        eval_dataset_class = DATASETS[dataset]
        dataset_config = config["morphing"]
    else:
        raise ValueError("Invalid dataset name.")

    # Load the dataset
    train_dataset = train_dataset_class(
        root_dir=config["data_dir"] + dataset_config["train_subdir"],
        protocol_file_name=dataset_config["train_protocol"],
        variant="train",
    )
    val_dataset = train_dataset_class(
        root_dir=config["data_dir"] + dataset_config["dev_subdir"],
        protocol_file_name=dataset_config["dev_protocol"],
        variant="dev",
    )
    if "2021DF" in dataset:
        eval_dataset = eval_dataset_class(
            root_dir=config["data_dir"] + dataset_config["eval_subdir"],
            protocol_file_name=dataset_config["eval_protocol"],
            variant="eval",
            local=True if "--local" in config["argv"] else False,
        )
    else:
        eval_dataset = eval_dataset_class(
            root_dir=config["data_dir"] + dataset_config["eval_subdir"],
            protocol_file_name=dataset_config["eval_protocol"],
            variant="eval",
        )

    # there is about 90% of spoofed recordings in the dataset, balance with weighted random sampling
    samples_weights = [train_dataset.get_class_weights()[i] for i in train_dataset.get_labels()]
    weighted_sampler = WeightedRandomSampler(samples_weights, len(train_dataset))

    bs = config["batch_size"] if not lstm else config["lstm_batch_size"]

    # create dataloader, use custom collate_fn to pad the data to the longest recording in batch
    collate_func = custom_single_batch_create if "single" in dataset else custom_pair_batch_create
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bs,
        collate_fn=collate_func,
        sampler=weighted_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=bs, collate_fn=collate_func, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=bs, collate_fn=collate_func, shuffle=True
    )

    return train_dataloader, val_dataloader, eval_dataloader
