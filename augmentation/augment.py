import torch
import torch_audiomentations as AA # This is what we use for noise and filter augmentations

from augmentation.general import GeneralAugmentations
from augmentation.codec import CodecAugmentations
from augmentation.RIR import RIRAugmentations
from augmentation.RawBoost import process_Rawboost_feature

def augment_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    Function to define the waveform augmentation pipeline.
    """
    pass
