from typing_extensions import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd
import numpy as np


def custom_batch_create(batch):
    # Get the lengths of all tensors in the batch
    lengths_gt = [item[0].size(1) for item in batch]
    lengths_test = [item[1].size(1) for item in batch]

    # Find the maximum length
    max_length_gt = max(lengths_gt)
    max_length_test = max(lengths_test)

    # Pad the tensors to have the maximum length
    padded_gts = []
    padded_tests = []
    labels = []
    for item in batch:
        waveform_gt = item[0]
        waveform_test = item[1]
        padded_waveform_gt = torch.nn.functional.pad(waveform_gt, (0, max_length_gt - waveform_gt.size(1)))
        padded_waveform_test = torch.nn.functional.pad(waveform_test, (0, max_length_test - waveform_test.size(1)))
        label = torch.tensor(item[2])

        padded_gts.append(padded_waveform_gt)
        padded_tests.append(padded_waveform_test)
        labels.append(label)

    padded_gts = torch.stack(padded_gts)
    padded_tests = torch.stack(padded_tests)
    labels = torch.stack(labels)

    return padded_gts, padded_tests, labels


class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_dir, protocol_file_name, variant: Literal["train", "dev", "eval"] = "train"):
        self.root_dir = root_dir  # Path to the LA folder

        protocol_file = os.path.join(self.root_dir, "ASVspoof2019_LA_cm_protocols", protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file, sep=" ", header=None)
        self.protocol_df.columns = ["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"]

        self.rec_dir = os.path.join(self.root_dir, "ASVspoof2019_LA_" + variant, "flac")

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        """
        Currently extracts the wav2vec2 embeddings before returning the data
        TODO: Extract the embeddings in the model instead of here with torch.no_grad()
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]  # Get the speaker ID

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, test_audio_file_name + ".flac")
        test_waveform, _ = load(test_audio_name)  # Load the tested speech

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == "bonafide" else 1  # 0 for genuine speech, 1 for spoofing speech

        # Get the genuine speech of the same speaker for differentiation
        speaker_recordings_df = self.protocol_df[self.protocol_df["SPEAKER_ID"] == speaker_id]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        # Get a random genuine speech of the speaker using sample()
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, gt_audio_file_name + ".flac")
        gt_waveform, _ = load(gt_audio_name)  # Load the genuine speech

        # print(f"Loaded GT:{gt_audio_name} and TEST:{test_audio_name}")
        return gt_waveform, test_waveform, label

    def get_labels(self) -> np.ndarray:
        """
        Returns an array of labels for the dataset, where 0 is genuine speech and 1 is spoofing speech
        Used for computing class weights for the loss function and weighted random sampling (see train.py)
        """
        return self.protocol_df["KEY"].map({"bonafide": 0, "spoof": 1}).to_numpy()

    def get_class_weights(self) -> np.ndarray:
        """Returns an array of class weights for the dataset, where 0 is genuine speech and 1 is spoofing speech"""
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        return torch.FloatTensor(class_weights)
