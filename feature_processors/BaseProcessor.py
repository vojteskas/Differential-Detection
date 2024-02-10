class BaseProcessor():
    """
    Base class for feature processors. Specific feature processors should inherit from this class.

    Feature processors are used to process features extracted from audio data before passing them through a classifier.
    """

    def __call__(self, features):
        raise NotImplementedError("Feature processor needs to implement the '__call__' method.")
