import torch
import torch.nn as nn

from torchaudio.pipelines import WAV2VEC2_XLSR_300M, WAV2VEC2_XLSR_1B, WAV2VEC2_XLSR_2B


class XLSR_300M(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        XLSR_300M model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_XLSR_300M.get_model()

        self.finetune = finetune

        self.transformer_layers = 24
        self.feature_size = 1024

    def extract_features(self, input_data):  # input_data shape: (batch_size, seq_len)
        """
        Extract features from audio data.

        param input_data: Audio data to extract features from of shape: (batch_size, seq_len)

        return: Features extracted from the audio data of shape:
                (24 (transformer layers), batch_size, time_frame, feature_size == 1024)
        """
        with torch.set_grad_enabled(self.finetune):
            # extract a list of 24 tensors, one for each transformer layer
            # each tensor has shape (batch_size, time_frame, feature_size == 1024)
            emb = self.model.extract_features(input_data)[0]  # [0] to get the features only

            # return as a single tensor with shape:
            # (24 (transformer layers), batch_size, time_frame, feature_size == 1024)
            return torch.stack(emb)


class XLSR_1B(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        XLSR_1B model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_XLSR_1B.get_model()

        self.finetune = finetune

        self.transformer_layers = 48
        self.feature_size = 1280

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


class XLSR_2B(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        XLSR_1B model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_XLSR_2B.get_model()

        self.finetune = finetune

        self.transformer_layers = 48
        self.feature_size = 1920

    def extract_features(self, input_data):  # input_data shape: (batch_size, seq_len)
        """
        Extract features from audio data.

        param input_data: Audio data to extract features from of shape: (batch_size, seq_len)

        return: Features extracted from the audio data of shape:
                (48 (transformer layers), batch_size, time_frame, feature_size == 1920)
        """
        with torch.set_grad_enabled(self.finetune):
            # extract a list of 48 tensors, one for each transformer layer
            # each tensor has shape (batch_size, time_frame, feature_size == 1920)
            emb = self.model.extract_features(input_data)[0]  # [0] to get the features only

            # return as a single tensor with shape:
            # (48 (transformer layers), batch_size, time_frame, feature_size == 1920)
            return torch.stack(emb)
