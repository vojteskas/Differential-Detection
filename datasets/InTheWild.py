import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio import load
import numpy as np


class InTheWildDataset_base(Dataset):
    """
    Base class for the InTheWild dataset. This class should not be used directly, but rather subclassed.
    The main subclasses are InTheWildDataset_pair and InTheWildDataset_single for providing pairs of
    genuine and spoofing speech for differential-based detecion and single recordings for "normal" detection,
    respectively.

    param root_dir: Path to the InTheWild folder
    param protocol_file_name: Name of the protocol file to use
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    """

    def __init__(self, root_dir, protocol_file_name="meta.csv", variant="eval"):
        self.root_dir = root_dir  # Path to the InTheWild folder

        # Headers from csv: file, speaker, label
        self.protocol_df = pd.read_csv(os.path.join(self.root_dir, protocol_file_name))

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in a specific subclass")

    def get_labels(self) -> np.ndarray:
        """
        Returns an array of labels for the dataset, where 0 is genuine speech and 1 is spoofing speech
        Used for computing class weights for the loss function and weighted random sampling (see train.py)
        """
        return self.protocol_df["label"].map({"bona-fide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self):
        """Returns an array of class weights for the dataset, where 0 is genuine speech and 1 is spoofing speech"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


class InTheWildDataset_pair(InTheWildDataset_base):
    """
    Dataset class for InTheWild that provides pairs of genuine and spoofing speech for differential-based detection.
    """

    def __init__(self, root_dir, protocol_file_name="meta.csv", variant="eval"):
        super().__init__(root_dir, protocol_file_name)

    def __getitem__(self, idx):
        """
        Returns tuples of the form (test_audio_file_name, gt_waveform, test_waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "speaker"]  # Get the speaker ID

        test_audio_file_name = str(self.protocol_df.loc[idx, "file"])
        test_waveform, _ = load(os.path.join(self.root_dir, test_audio_file_name))

        # 0 for genuine speech, 1 for spoofing speech
        label = 0 if self.protocol_df.loc[idx, "label"] == "bona-fide" else 1

        # Get the genuine speech of the same speaker for differentiation
        speaker_recordings_df = self.protocol_df[
            (self.protocol_df["speaker"] == speaker_id) & (self.protocol_df["label"] == "bona-fide")
        ]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        # Get a random genuine speech from the same speaker
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["file"]
        gt_waveform, _ = load(os.path.join(self.root_dir, gt_audio_file_name))

        return test_audio_file_name, gt_waveform, test_waveform, label


class InTheWildDataset_single(InTheWildDataset_base):
    """
    Dataset class for InTheWild that provides single recordings for "normal" detection.
    """

    def __init__(self, root_dir, protocol_file_name="meta.csv", variant="eval"):
        super().__init__(root_dir, protocol_file_name)

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = str(self.protocol_df.loc[idx, "file"])
        waveform, _ = load(os.path.join(self.root_dir, audio_file_name))

        # 0 for genuine speech, 1 for spoofing speech
        label = 0 if self.protocol_df.loc[idx, "label"] == "bona-fide" else 1

        return audio_file_name, waveform, label
