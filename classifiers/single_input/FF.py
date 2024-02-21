import torch.nn as nn
import torch.nn.functional as F

# TODO: Add dropout and regularization?
# TODO: Migrate to PyTorch Lightning?


class FF(nn.Module):
    """
    Linear classifier for audio embeddings.
    """

    def __init__(self, extractor, feature_processor, in_dim=1024):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        param in_dim: Dimension of the input data to the classifier, divisible by 4.
        """

        super().__init__()

        self.extractor = extractor
        self.feature_processor = feature_processor

        # Allow variable input dimension, mainly for base (768 features) and large models (1024 features)
        self.layer1_in_dim = in_dim
        self.layer1_out_dim = in_dim // 2
        self.layer2_in_dim = self.layer1_out_dim
        self.layer2_out_dim = self.layer2_in_dim // 2

        # Experiment with LayerNorm instead of BatchNorm
        self.classifier = nn.Sequential(
            nn.Linear(self.layer1_in_dim, self.layer1_out_dim),
            nn.BatchNorm1d(self.layer1_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_in_dim, self.layer2_out_dim),
            nn.BatchNorm1d(self.layer2_out_dim),
            nn.ReLU(),
            nn.Linear(self.layer2_out_dim, 2),  # output 2 classes
        )

    def forward(self, waveforms):
        """
        Forward pass through the model.

        Extract features from the audio data, process them and pass them through the classifier.

        param embeddings: Audio waveforms of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb = self.extractor.extract_features(waveforms)

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob
