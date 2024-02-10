from BaseProcessor import BaseProcessor


class MHFAProcessor(BaseProcessor):
    """
    Feature processor implementing Multi-head factorized attentive pooling.
    """
    def __init__(self, num_heads: int = 4, dim: int = 0):
        """
        Initialize the feature processor.

        param num_heads: Number of heads to use
        param dim: Dimension to pool over
        """
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim

    def __call__(self, features):
        """
        Process features extracted from audio data - Multi-head factorized attentive pooling.

        param features: Features extracted from the audio data

        return: Processed features without the pooled dimension
        """

        # TODO: implement MHFA pooling
        raise NotImplementedError("MHFA pooling not implemented yet.")
