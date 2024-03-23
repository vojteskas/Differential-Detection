import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add dropout and regularization?
# TODO: Migrate to PyTorch Lightning?


class FFDiffBase(nn.Module):
    """
    Base class for linear classifiers which use the difference between two embeddings for classification.
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


class FFDiff(FFDiffBase):
    """
    Linear classifier using subtraction as a difference metric.
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

        diff = emb_gt - emb_test

        out = self.classifier(diff)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFDiff2(FFDiffBase):
    """
    Linear classifier using subtraction as a difference metric.
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

        diff = emb_gt - emb_test

        emb = self.feature_processor(diff)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob
    

class FFDiffAbs(FFDiffBase):
    """
    Linear classifier using the absolute value of subtraction as a difference metric.
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

        diff = torch.abs(emb_gt - emb_test)

        out = self.classifier(diff)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFDiffAbs2(FFDiffBase):
    """
    Linear classifier using subtraction as a difference metric.
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

        diff = torch.abs(emb_gt - emb_test)

        emb = self.feature_processor(diff)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFDiffQuadratic(FFDiffBase):
    """
    Linear classifier using the absolute value of subtraction as a difference metric.
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

        diff = torch.pow(emb_gt - emb_test, 2)

        out = self.classifier(diff)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFDiffQuadratic2(FFDiffBase):
    """
    Linear classifier using subtraction as a difference metric.
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

        diff = torch.pow(emb_gt - emb_test, 2)

        emb = self.feature_processor(diff)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob
