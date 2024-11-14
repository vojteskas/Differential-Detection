import torch

from augmentation.general import change_speed

def augment_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    Function to define the waveform augmentation pipeline.
    """
    # Change the speed of the audio
    speed_factor = 1.1
    waveform = change_speed(waveform, speed_factor)

    return waveform
