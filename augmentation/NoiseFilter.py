import audiomentations as AA
import numpy
import torch


class NoiseFilterAugmentations:
    """
    Class for noise and filter augmentations.
    Uses audiomentations (currently cpu-only; torch-audiomentations doesnt support what we need at the moment).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        # Noises
        self.color_noise = AA.AddColorNoise(p=1.0)
        self.gaussian_noise = AA.AddGaussianNoise(p=1.0)
        self.gaussian_snr = AA.AddGaussianSNR(p=1.0)

        # Filters
        self.band_pass_filter = AA.BandPassFilter(p=1.0)
        self.band_stop_filter = AA.BandStopFilter(p=1.0)
        self.high_pass_filter = AA.HighPassFilter(p=1.0)
        self.high_shelf_filter = AA.HighShelfFilter(p=1.0)
        self.low_pass_filter = AA.LowPassFilter(p=1.0)
        self.low_shelf_filter = AA.LowShelfFilter(p=1.0)
        self.peaking_filter = AA.PeakingFilter(p=1.0)

        # Compose to a single augmentation
        # Using one of each noise and filter augmentation according to https://www.isca-archive.org/asvspoof_2024/xu24_asvspoof.pdf
        self.augment = AA.Compose(
            [
                AA.OneOf(
                    [
                        self.color_noise,
                        self.gaussian_noise,
                        self.gaussian_snr,
                    ],
                    p=0.9,
                ),
                AA.OneOf(
                    [
                        self.band_pass_filter,
                        self.band_stop_filter,
                        self.high_pass_filter,
                        self.high_shelf_filter,
                        self.low_pass_filter,
                        self.low_shelf_filter,
                        self.peaking_filter,
                    ],
                    p=0.9,
                ),
            ]
        )

    def apply_noise_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply noise and filter augmentations to the audio waveform.

        param waveform: The audio waveform to apply noise filter augmentations to.

        return: The audio waveform with noise filter augmentations applied.
        """
        numpy_waveform = waveform.squeeze()
        if not isinstance(
            numpy_waveform, numpy.ndarray
        ):  # Convert from tensor to numpy array if not already
            numpy_waveform = numpy_waveform.cpu().numpy()
        augmented_waveform = self.augment(samples=numpy_waveform, sample_rate=self.sample_rate)
        return torch.tensor(augmented_waveform)
