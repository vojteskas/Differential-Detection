from typing import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd
import numpy as np


# Consider doing own train/val split, now its 50/50, like 80/20 should suffice and give more training data
class ASVspoof2019LADataset_base(Dataset):
    """
    Base class for the ASVspoof2019LA dataset. This class should not be used directly, but rather subclassed.
    The main subclasses are ASVspoof2019LADataset_pair and ASVspoof2019LADataset_single for providing pairs of
    genuine and spoofing speech for differential-based detecion and single recordings for "normal" detection,
    respectively.

    param root_dir: Path to the ASVspoof2019LA folder
    param protocol_file_name: Name of the protocol file to use
    param variant: One of "train", "dev", "eval" to specify the dataset variant
    """

    def __init__(self, root_dir, protocol_file_name, variant: Literal["train", "dev", "eval"] = "train"):
        self.root_dir = root_dir  # Path to the LA folder

        protocol_file = os.path.join(self.root_dir, "ASVspoof2019_LA_cm_protocols", protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file, sep=" ", header=None)
        self.protocol_df.columns = ["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"]

        self.rec_dir = os.path.join(self.root_dir, f"ASVspoof2019_LA_{variant}", "flac")

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in a specific subclass")

    def get_labels(self) -> np.ndarray:
        """
        Returns an array of labels for the dataset, where 0 is genuine speech and 1 is spoofing speech
        Used for computing class weights for the loss function and weighted random sampling (see train.py)
        """
        return self.protocol_df["KEY"].map({"bonafide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self):
        """Returns an array of class weights for the dataset, where 0 is genuine speech and 1 is spoofing speech"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)


class ASVspoof2019LADataset_pair(ASVspoof2019LADataset_base):
    def __init__(self, root_dir, protocol_file_name, variant: Literal["train", "dev", "eval"] = "train"):
        super().__init__(root_dir, protocol_file_name, variant)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]  # Get the speaker ID

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, f"{test_audio_file_name}.flac")
        test_waveform, _ = load(test_audio_name)  # Load the tested speech

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == "bonafide" else 1  # 0 for genuine speech, 1 for spoofing speech

        # Get the genuine speech of the same speaker for differentiation
        speaker_recordings_df = self.protocol_df[self.protocol_df["SPEAKER_ID"] == speaker_id]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        # Get a random genuine speech of the speaker using sample()
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, f"{gt_audio_file_name}.flac")
        gt_waveform, _ = load(gt_audio_name)  # Load the genuine speech

        # print(f"Loaded GT:{gt_audio_name} and TEST:{test_audio_name}")
        return gt_waveform, test_waveform, label


class ASVspoof2019LADataset_single(ASVspoof2019LADataset_base):
    def __init__(self, root_dir, protocol_file_name, variant: Literal["train", "dev", "eval"] = "train"):
        super().__init__(root_dir, protocol_file_name, variant)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        audio_name = os.path.join(self.rec_dir, f"{audio_file_name}.flac")
        waveform, _ = load(audio_name)

        # 0 for genuine speech, 1 for spoofing speech
        label = 0 if self.protocol_df.loc[idx, "KEY"] == "bonafide" else 1

        return waveform, label
