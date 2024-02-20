import torch
import torch.nn as nn

from torchaudio.pipelines import HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE


class HuBERT_base(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        HuBERT base model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = HUBERT_BASE.get_model()

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


class HuBERT_large(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        HuBERT large model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = HUBERT_LARGE.get_model()

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


class HuBERT_extralarge(nn.Module):
    def __init__(self, finetune: bool = False):
        """
        HuBERT extra large model for extracting features from audio data.

        param finetune: Whether to allow the model to be finetuned.
        """
        super().__init__()

        self.model = HUBERT_XLARGE.get_model()

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
