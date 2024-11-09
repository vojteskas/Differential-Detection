import torch
import torch.nn as nn
import torch.nn.functional as F

from classifiers.FFBase import FFBase
from feature_processors.AASIST import AASIST


class FFConcatBase(FFBase):
    """
    Base class for feedforward classifiers which concatenate tested and ground truth recording for classification.
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


class FFConcat1(FFConcatBase):
    """
    Feedforward classifier which concatenates tested and ground truth recording for classification.

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
        input_data = torch.cat(
            (input_data_ground_truth, input_data_tested), 1
        )  # Concatenate along the time axis

        emb = self.extractor.extract_features(input_data)

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFConcat2(FFConcatBase):
    """
    Feedforward classifier which concatenates tested and ground truth recording for classification.

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
        emb = torch.cat((emb_gt, emb_test), 2)  # Concatenate along the time axis

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFConcat3(FFConcatBase):
    """
    Feedforward classifier which concatenates tested and ground truth recording for classification.

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

        super().__init__(
            extractor, feature_processor, in_dim * 2
        )  # Double the input dimension because concat

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
        emb = torch.cat((emb_gt, emb_test), 1)  # Concatenate along the feature axis (1), not batch axis (0)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFLSTM(FFConcatBase):
    """
    Feedforward classifier which concatenates tested and ground truth recording for classification.

    After feature extraction and average pooling, the embeddings are passed through an LSTM to find the differences between the embeddings.
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

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=in_dim,
            num_layers=2,
            batch_first=True,
        )

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

        if type(self.feature_processor) is AASIST:
            emb = torch.cat((emb_gt, emb_test), 2)
            emb = emb[-1]

            # LSTM
            emb, _ = self.lstm(emb)

            # AASIST
            emb = self.feature_processor(emb)
        else:
            emb_gt = torch.mean(emb_gt, 0)
            emb_test = torch.mean(emb_test, 0)
            emb = torch.cat((emb_gt, emb_test), 1) # emb in shape (batch, seq_len, feature_size)
        
            # LSTM
            emb, _ = self.lstm(emb)

            emb = emb[:, -1, :]  # Take the last hidden state

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob


class FFLSTM2(FFConcatBase):
    """
    Feedforward classifier which concatenates tested and ground truth recording for classification.

    After feature extraction, the embeddings are concatenated and flattened.
    Then, they are passed through an LSTM to find the differences between the embeddings.
    Finally, the embeddings are unflattened back to the original shape before feature processing and classification.
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

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=in_dim,
            num_layers=2,
            batch_first=True,
        )

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
        
        emb = torch.cat((emb_gt, emb_test), 2)
        translayers, _, totalframes, _ = emb.size() # needed for unflattening later

        if type(self.feature_processor) is AASIST:
            emb = emb[-1]
        else:
            emb = emb.transpose(0, 1).flatten(1, 2)

        # LSTM 
        emb, _ = self.lstm(emb)

        # transform back to original shape (translayers, batch, totalframes, feature_size)
        if type(self.feature_processor) is not AASIST:
            emb = emb.unflatten(1, (translayers, totalframes)).transpose(0, 1)

        emb = self.feature_processor(emb)

        out = self.classifier(emb)
        prob = F.softmax(out, dim=1)

        return out, prob
