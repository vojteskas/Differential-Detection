import torch
import torch.nn as nn

from torchaudio.pipelines import WAV2VEC2_XLSR_1B

class XLSR_1B(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        XLSR_1B model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_XLSR_1B.get_model()

        self.finetune = finetune

    def extract_features(self, input_data):  # input_data shape: (batch_size, seq_len)
        """
        Extract features from audio data.

        param input_data: Audio data to extract features from of shape: (batch_size, seq_len)

        return: Features extracted from the audio data of shape:
                (48 (transformer layers), batch_size, time_frame, feature_size == 1280)
        """
        with torch.set_grad_enabled(self.finetune):
            # extract a list of 48 tensors, one for each transformer layer
            # each tensor has shape (batch_size, time_frame, feature_size == 1280)
            emb = self.model.extract_features(input_data)[0]  # [0] to get the features only

            # return as a single tensor with shape:
            # (48 (transformer layers), batch_size, time_frame, feature_size == 1280)
            return torch.stack(emb)
