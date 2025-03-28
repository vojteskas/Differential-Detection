import torch
import torch.nn as nn
import torch.nn.functional as F


class FFDot(nn.Module):
    """
    WARNING: Only for testing and demonstration purposes. This classifier does not make sense and does not work (which is expected).

    Feedforward classifier using the dot product as a difference metric.
    """

    def __init__(self, extractor, feature_processor):
        """
        Initialize the model.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        param feature_processor: Model to process the extracted features.
                                 Needs to provide method __call__(input_data)
        """

        super().__init__()

        self.extractor = extractor
        self.feature_processor = feature_processor

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass through the model.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: probability of the two audio files being the same speaker
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.feature_processor(emb_gt)
        emb_test = self.feature_processor(emb_test)

        # Calculate the dot product
        # The following batch matrix multiplication (bmm) is equivalent to the vector-wise dot product
        # Faster equivalent to
        # for v1, v2 in zip(emb_gt, emb_test):
        #     torch.dot(v1, v2)
        score = torch.bmm(emb_gt.unsqueeze(1), emb_test.unsqueeze(2)).squeeze()

        prob = F.sigmoid(score)

        return prob
