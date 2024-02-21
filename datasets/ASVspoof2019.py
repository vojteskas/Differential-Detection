from typing import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd
import numpy as np


def custom_pair_batch_create(batch: list):
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths_gt = torch.tensor([item[0].size(1) for item in batch])
    lengths_test = torch.tensor([item[1].size(1) for item in batch])

    # Find the maximum length
    max_length_gt = int(torch.max(lengths_gt))
    max_length_test = int(torch.max(lengths_test))

    # Pad the tensors to have the maximum length
    padded_gts = torch.zeros(batch_size, max_length_gt)
    padded_tests = torch.zeros(batch_size, max_length_test)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        waveform_gt = item[0]
        waveform_test = item[1]
        padded_waveform_gt = torch.nn.functional.pad(
            waveform_gt, (0, max_length_gt - waveform_gt.size(1))
        ).squeeze(0)
        padded_waveform_test = torch.nn.functional.pad(
            waveform_test, (0, max_length_test - waveform_test.size(1))
        ).squeeze(0)
        label = torch.tensor(item[2])

        padded_gts[i] = padded_waveform_gt
        padded_tests[i] = padded_waveform_test
        labels[i] = label

    return padded_gts, padded_tests, labels

def custom_single_batch_create(batch: list):
    # Free unused memory before creating the new batch
    # This is necessary because PyTorch has trouble with dataloader memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Get the lengths of all tensors in the batch
    batch_size = len(batch)
    lengths = torch.tensor([item[0].size(1) for item in batch])

    # Find the maximum length
    max_length = int(torch.max(lengths))

    # Pad the tensors to have the maximum length
    padded_waveforms = torch.zeros(batch_size, max_length)
    labels = torch.zeros(batch_size)
    for i, item in enumerate(batch):
        waveform = item[0]
        padded_waveform = torch.nn.functional.pad(
            waveform, (0, max_length - waveform.size(1))
        ).squeeze(0)
        label = torch.tensor(item[1])

        padded_waveforms[i] = padded_waveform
        labels[i] = label

    return padded_waveforms, labels


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
