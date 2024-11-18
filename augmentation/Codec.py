import torch
import torchaudio.transforms as T


class CodecAugmentations:
    """
    Class for codec augmentations.
    Currently supports mu-law compression.
    """

    def __init__(self, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = sample_rate
        self.mu_encoder = T.MuLawEncoding().to(self.device)
        self.mu_decoder = T.MuLawDecoding().to(self.device)

    def mu_law(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mu-law compression to the audio waveform.

        param waveform: The audio waveform to apply mu-law compression to.

        return: The audio waveform with mu-law compression and decompression applied.
        """
        waveform = waveform.to(self.device)
        enc = self.mu_encoder(waveform)
        dec = self.mu_decoder(enc)
        return dec

    def mp3(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mp3 compression to the audio waveform.

        param waveform: The audio waveform to apply mp3 compression to.

        return: The audio waveform with mp3 compression applied.
        """
        raise NotImplementedError(
            "MP3 compression not yet implemented."
        )  # Blame torchaudio for not having mp3 compression, maybe try audiomentations
