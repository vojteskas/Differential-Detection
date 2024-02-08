from typing import Literal
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add dropout and regularization?
# TODO: Migrate to PyTorch Lightning?


class FFDiff(nn.Module):
    def __init__(self, extractor):
        """
        Linear classifier using the difference between two embeddings.

        param extractor: Model to extract features from audio data.
                         Needs to provide method extract_features(input_data)
        """

        super().__init__()

        self.extractor = extractor

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, input_data_ground_truth, input_data_tested):
        """
        Forward pass through the model.

        Extract features from the audio data, process them (pooling) and pass them through the classifier.

        param input_data_ground_truth: Audio data of the ground truth of shape: (batch_size, seq_len)
        param input_data_tested: Audio data of the tested data of shape: (batch_size, seq_len)

        return: Output of the model (logits) and the probabilities (softmax output of the logits).
        """

        emb_gt = self.extractor.extract_features(input_data_ground_truth)
        emb_test = self.extractor.extract_features(input_data_tested)

        emb_gt = self.process_features(emb_gt)
        emb_test = self.process_features(emb_test)

        diff = emb_gt - emb_test

        out = self.classifier(diff)
        prob = F.softmax(out, dim=1)

        return out, prob

    def process_features(self, features, pooling: Literal["mean", "mhfa"] = "mean"):
        """
        Process features extracted from audio data.

        param features: Features extracted from the audio data of shape:
                        (num_transformer_layers or 1, batch_size, time_frame, feature_size == 1024)
        param pooling: Pooling type to use for processing

        return: Processed features of shape: (batch_size, 1024)
        """

        # TODO: add MHFA pooling as an alternative

        if pooling == "mean":
            # return the mean of the features over the time frame and transformer layers
            return features.mean(dim=2).mean(dim=0)
        else:
            raise NotImplementedError(f"Pooling type '{pooling}' not supported.")
