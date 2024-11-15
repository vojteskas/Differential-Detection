import torch
import torchaudio.transforms as T


class CodecAugmentations:
    """
    Class for codec augmentations.
    Currently supports mu-law and mp3 compression.
    """
    def __init__(self, sample_rate: int = 16000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sample_rate = sample_rate
        mu_encoder = T.MuLawEncoding().to(self.device)
        mu_decoder = T.MuLawDecoding().to(self.device)

    def mu_law(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply mu-law compression to the audio waveform.

        param waveform: The audio waveform to apply mu-law compression to.
        param quantization_channels: The number of quantization channels.

        return: The audio waveform with mu-law compression applied.
        """
        enc = T.MuLawEncoding()(waveform)
        dec = T.MuLawDecoding()(enc)
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
        raise NotImplementedError("MP3 compression not yet implemented.") # Blame torchaudio for not having mp3 compression
