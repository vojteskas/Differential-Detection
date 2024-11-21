#!/usr/bin/env python3
from typing import Literal
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T


class RIRDataset:
    """
    Class for RIR augmentation. Contains a pandas dataframe with the filepaths. Can be randomly sampled from.
    """

    def __init__(self, rir_root):
        self.rir_root = rir_root
        # Pointsource noises are the sounds and not the impulse responses
        pointsource_df = pd.read_csv(
            rir_root + "RIRS_NOISES/pointsource_noises/noise_list", sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths
        isotropic_df = pd.read_csv(
            rir_root + "RIRS_NOISES/real_rirs_isotropic_noises/noise_list", sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths
        rir_df = pd.read_csv(
            rir_root + "RIRS_NOISES/real_rirs_isotropic_noises/rir_list", sep=" ", header=None
        ).iloc[:, -1]  # Get the last column that contains the filepaths

        # Remove RWCP from the isotropic noises (is not mono)
        self.df = pd.concat([pointsource_df, isotropic_df, rir_df], ignore_index=True)
        self.df = self.df[~self.df.str.contains("RWCP")]

    def __len__(self):
        return len(self.df)

    def get_random_rir(self):
        path = self.rir_root + self.df.sample(1).iloc[0]
        # print(f"Loading RIR from {path}.")
        # path = "/mnt/d/VUT/Deepfakes/Datasets/RIR/RIRS_NOISES/real_rirs_isotropic_noises/RVB2014_type2_noise_simroom1_2.wav"
        try:
            rir = torchaudio.load(path)[0]
        except Exception as e:
            print(f"Failed to load RIR from {path}.")
            raise e
        if rir.size(0) > 1:
            rir = rir.mean(0)
        return rir.squeeze()


class RIRAugmentations:
    """
    Class for RIR augmentations.
    """

    def __init__(
        self, rir_root: str, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.rir_dataset = RIRDataset(rir_root)
        self.convolver = T.FFTConvolve(mode="same").to(self.device)

    def apply_rir(
        self,
        waveform: torch.Tensor,
        method: Literal["convolve", "superimpose"] = "superimpose",
        scale_factor: float = 0.5,
    ) -> torch.Tensor:
        """
        Apply a random RIR to the audio waveform.

        param waveform: The audio waveform to apply the RIR to.
        param method: The method to apply the RIR with. Can be "convolve" or "superimpose".
        param scale_factor: The scale factor to apply to the RIR, should be between 0.2 and 0.8.

        return: The audio waveform with the RIR applied.
        """
        waveform = waveform.to(self.device)
        rir = self.rir_dataset.get_random_rir().to(self.device)
        if method == "convolve":
            raise NotImplementedError("Convolution is not working as expected, consider not implemented.")
            wf = self.convolver(waveform, rir * scale_factor)
            return wf / torch.max(torch.abs(wf))
        elif method == "superimpose":
            # Cut or pad the RIR to match the waveform length
            if len(rir) > len(waveform):
                rir = rir[: len(waveform)]
            elif len(rir) < len(waveform):
                rir = torch.nn.functional.pad(rir, (0, len(waveform) - len(rir)))
            wf = waveform + rir * scale_factor
            return wf / torch.max(torch.abs(wf))


if __name__ == "__main__":
    rir_augment = RIRAugmentations("/mnt/d/VUT/Deepfakes/Datasets/RIR/")
    waveform, sr = torchaudio.load("real.wav")
    waveform = waveform.squeeze()
    while True:
        print(f"Before convolution: {waveform.shape}")
        augmented_waveform = rir_augment.apply_rir(waveform, method="superimpose", scale_factor=0.8)
        print(f"After convolution: {augmented_waveform.shape}")
        torchaudio.save("convolved.wav", augmented_waveform.unsqueeze(0), sr)
        break
