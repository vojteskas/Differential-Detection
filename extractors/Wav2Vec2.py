import torch
import torch.nn as nn

from torchaudio.pipelines import WAV2VEC2_BASE, WAV2VEC2_LARGE, WAV2VEC2_LARGE_LV60K


class Wav2Vec2_base(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        Wav2Vec 2.0 base model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_BASE.get_model()

        self.finetune = finetune

        self.transformer_layers = 12
        self.feature_size = 768

    def extract_features(self, input_data):  # input_data shape: (batch_size, seq_len)
        """
        Extract features from audio data.

        param input_data: Audio data to extract features from of shape: (batch_size, seq_len)

        return: Features extracted from the audio data of shape:
                (12 (transformer layers), batch_size, time_frame, feature_size == 768)
        """
        with torch.set_grad_enabled(self.finetune):
            # extract a list of 12 tensors, one for each transformer layer
            # each tensor has shape (batch_size, time_frame, feature_size == 768)
            emb = self.model.extract_features(input_data)[0]  # [0] to get the features only

            # return as a single tensor with shape:
            # (12 (transformer layers), batch_size, time_frame, feature_size == 768)
            return torch.stack(emb)


class Wav2Vec2_large(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        Wav2Vec 2.0 large model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_LARGE.get_model()

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


class Wav2Vec2_LV60k(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        Wav2Vec 2.0 LV60k model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = WAV2VEC2_LARGE_LV60K.get_model()

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
