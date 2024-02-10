from typing import Union, Sequence

from feature_processors.BaseProcessor import BaseProcessor


class MeanProcessor(BaseProcessor):
    """
    Feature processor implementing average pooling over a dimension.
    """

    def __init__(self, dim: Union[int, Sequence[int]] = (0, 2)):
        """
        Initialize the feature processor.

        param dim: Dimension(s) to pool over
        """
        super().__init__()

        self.dim = dim

    def __call__(self, features):
        """
        Process features extracted from audio data - average pooling over dimension(s).

        param features: Features extracted from the audio data

        return: Processed features without the pooled dimension(s) specified in the constructor
        """

        return features.mean(dim=self.dim)
