from torch import cat
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add dropout and regularization?
# TODO: Migrate to PyTorch Lightning?


class FFConcatBase(nn.Module):
    """
    Base class for linear classifiers which concatenate tested and ground truth recording for classification.
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

    def forward(self, input_gt, input_tested):
        raise NotImplementedError("Forward pass not implemented in the base class.")


class FFConcat1(FFConcatBase):
    """
    Linear classifier which concatenates tested and ground truth recording for classification.

    Concatenation happens before feature extraction.
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
        super().__init__(extractor, feature_processor, in_dim)

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass through the model.

        Extract features from the audio data, process them and pass them through the classifier.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        # Concat
        input_data = cat((input_data_ground_truth, input_data_tested), 1)  # Concatenate along the time axis

        emb = self.extractor.extract_features(input_data)

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFConcat2(FFConcatBase):
    """
    Linear classifier which concatenates tested and ground truth recording for classification.

    Concatenation happens after feature extraction but before feature processing.
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

        super().__init__(extractor, feature_processor, in_dim)

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass through the model.

        Extract features from the audio data, process them and pass them through the classifier.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        # Concat
        emb = cat((emb_gt, emb_test), 2)  # Concatenate along the time axis

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFConcat3(FFConcatBase):
    """
    Linear classifier which concatenates tested and ground truth recording for classification.

    Concatenation happens after feature processing.
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

        super().__init__(extractor, feature_processor, in_dim)

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass through the model.

        Extract features from the audio data, process them and pass them through the classifier.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the class probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        # Concat
        emb = cat((emb_gt, emb_test), 1)  # Concatenate along the feature axis (1), not batch axis (0)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFLSTM(FFConcatBase):
    def __init__(self, extractor, feature_processor, in_dim=1024):
        super().__init__(extractor, feature_processor, in_dim)

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=in_dim,
            num_layers=2,
            batch_first=True,
        )

    def forward(self, input_data_ground_truth, input_data_tested):
        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = torch.mean(emb_gt, 0)
        emb_test = torch.mean(emb_test, 0)

        # emb in shape (batch, seq_len, feature_size)
        # LSTM to find the differences between emb_gt and emb_test
        emb = cat((emb_gt, emb_test), 1)
        emb, _ = self.lstm(emb)
        
        emb = emb[:, -1, :]  # Take the last hidden state

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob
