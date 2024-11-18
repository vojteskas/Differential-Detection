from math import floor
from typing import Literal, Tuple
import torch
import torchaudio.transforms as T


class GeneralAugmentations:
    """
    Class for general augmentations.
    Currently supports speed, volume, time masking, trimming starting silence.
    """

    def __init__(self, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = sample_rate  # 16kHz is the default
        self.spectrogram_factory = T.Spectrogram(power=None).to(self.device)
        self.voice_activity_detector = T.Vad(sample_rate=self.sample_rate).to(self.device)
        self.time_stretcher = T.TimeStretch().to(self.device)

    def change_speed_pitch(
        self,
        waveform: torch.Tensor,
        speed_factor: float,
    ) -> torch.Tensor:
        """
        Change the **speed** (not length) of the audio by a given factor while also changing the pitch.

        param waveform: The audio waveform to change the speed of.
        param speed_factor: The factor to change the speed by.

        return: The audio waveform with the speed changed.
        """
        return T.Resample(orig_freq=self.sample_rate, new_freq=int(self.sample_rate / speed_factor))(waveform)

    def change_speed(  # Sounds very robotic, suggesting to not use
        self,
        waveform: torch.Tensor,
        speed_factor: float,
    ) -> torch.Tensor:
        """
        Change the **speed** (not length) of the audio by a given factor without changing the pitch.

        param waveform: The audio waveform to change the speed of.
        param speed_factor: The factor to change the speed by.
        """
        waveform = waveform.to(self.device)
        spectrogram = self.spectrogram_factory(waveform)
        stretched = self.time_stretcher(spectrogram, speed_factor)
        return T.InverseSpectrogram()(stretched)

    def change_volume(
        self,
        waveform: torch.Tensor,
        volume_factor: float,
        gain_type: Literal["amplitude", "power", "db"] = "amplitude",
    ) -> torch.Tensor:
        """
        Change the volume of the audio by a given factor.

        param waveform: The audio waveform to change the volume of.
        param volume_factor: The factor to change the volume by.
            If gain_type = amplitude, gain is a positive amplitude ratio. I.e., 0.5 will halve the absolute (not percieved) volume.
            If gain_type = power, gain is a power (voltage squared). I.e., 0.5 will quarter the absolute (not percieved) volume.
            If gain_type = db, gain is in decibels. I.e., 3db is doubling of power, 6db is doubling of amplitude/voltage, 10db is doubling of human-perceived loudness.
        param gain_type: The type of gain to apply to the audio. Can be "amplitude", "power", or "db".

        return: The audio waveform with the volume changed.
        """
        return T.Vol(volume_factor, gain_type)(waveform)

    def mask_time(  # lenght should be 20-50% and position should be random
        self,
        waveform: torch.Tensor,
        mask_time: Tuple[float, float],
        selection: Literal["samples", "time"] = "samples",
    ) -> torch.Tensor:
        """
        Mask the audio in the time domain.

        param waveform: The audio waveform to mask.
        param mask_time: Tuple of the start and end time to mask in seconds.

        return: The audio waveform with the mask applied.
        """
        if selection == "time":
            waveform[floor(mask_time[0] * self.sample_rate) : floor(mask_time[1] * self.sample_rate)] = 0.0
        elif selection == "samples":
            waveform[floor(mask_time[0]) : floor(mask_time[1])] = 0.0
        return waveform 
    
    def trim_starting_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply voice activity detection to the audio.

        param waveform: The audio waveform to apply voice activity detection to.

        return: The audio waveform with only the voice detected.
        """
        return self.voice_activity_detector(waveform)
