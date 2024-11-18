from math import floor
import torch
import random

from augmentation.Codec import CodecAugmentations
from augmentation.General import GeneralAugmentations
from augmentation.NoiseFilter import NoiseFilterAugmentations
from augmentation.RawBoost import process_Rawboost_feature
from augmentation.RIR import RIRAugmentations


class Augmentor:
    """
    Class to define the waveform augmentation pipeline.
    """

    def __init__(self, rir_root):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.Codec = CodecAugmentations(device=self.device)
        self.General = GeneralAugmentations(device=self.device)
        self.NoiseFilter = NoiseFilterAugmentations()  # is cpu only as of now
        self.RIR = RIRAugmentations(rir_root, device=self.device)

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        The waveform augmentation pipeline.
        """

        # Using augmentations according to this paper: https://www.isca-archive.org/asvspoof_2024/xu24_asvspoof.pdf
        trim_starting_silence: bool = random.random() < 0.5  # 50% chance of removing the starting silence
        apply_rir_timemask: bool = random.random() < 0.3  # 30% chance of applying RIR augmentations
        apply_mu_law: bool = random.random() < 0.3  # 30% chance of applying mu-law enc-dec augmentation
        apply_LnL_ISD: bool = random.random() < 0.3  # 30% chance of applying RawBoost augmentations
        apply_noise_filter: bool = random.random() < 0.3  # 30% chance of applying noise augmentations

        waveform.squeeze()

        if trim_starting_silence:  # 50% chance of removing the starting silence
            waveform = self.General.trim_starting_silence(waveform)

        if apply_rir_timemask:
            wf_len = len(waveform)
            time_mask_duration = random.uniform(floor(wf_len * 0.2), floor(wf_len * 0.5))
            time_mask_start = random.uniform(0, wf_len - time_mask_duration)
            waveform = self.General.mask_time(
                waveform, mask_time=(time_mask_start, time_mask_start + time_mask_duration)
            )

            rir_intesity = random.uniform(0.2, 0.8)
            waveform = self.RIR.apply_rir(waveform, method="convolve", scale_factor=rir_intesity)

        if apply_mu_law:
            waveform = self.Codec.mu_law(waveform)

        if apply_LnL_ISD:
            waveform = process_Rawboost_feature(waveform, 16000, algo=5)

        if apply_noise_filter:
            waveform = self.NoiseFilter.apply_noise_filter(waveform)

        return waveform
