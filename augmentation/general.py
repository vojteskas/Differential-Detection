"""
File for general augmentations, such as speed, pitch, time-masking, etc.
"""

import librosa

import torch
import torchaudio as ta
import torchaudio.transforms as T

device="cuda" if torch.cuda.is_available() else "cpu"
sample_rate=16000
speed_perturbation = ta.transforms.SpeedPerturbation(16000, [0.9, 1.0, 1.1]).to("cuda")

def change_speed(
    waveform: torch.Tensor,
    speed_factor: float,
    pitch: bool = True,
) -> torch.Tensor:
    """
    Change the speed of the audio by a given factor.

    param waveform: The audio waveform to change the speed of.
    param speed_factor: The factor to change the speed by.
    param pitch: Whether to change the pitch of the audio as well.
    param sample_rate: The sample rate of the audio. Default is 16kHz.

    return: The audio waveform with the speed changed.
    """
    if pitch:
        waveform = waveform.to(device)
        transformed_waveform = speed_perturbation(waveform)
        return transformed_waveform
    else:
        return torch.from_numpy(librosa.effects.time_stretch(waveform.numpy(), rate=speed_factor))
