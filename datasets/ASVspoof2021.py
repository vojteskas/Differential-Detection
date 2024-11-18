from typing import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd
import numpy as np


class ASVspoof2021_base(Dataset):
    """
    Base class for the ASVspoof2021 LA and DF eval datasets. This class should not be used directly, but
    rather subclassed.

    param root_dir: Path to the ASVspoof2021{LA,DF} folder
    param protocol_file_name: Name of the protocol file to use
    param augment: Ignored, only for compatibility with other datasets (ASVspoof2021 is not a training dataset, therefore no augmentations)
    param rir_root: Ignored, only for compatibility with other datasets (ASVspoof2021 is not a training dataset, therefore no RIR augmentations)
    """

    def __init__(self, root_dir, protocol_file_name, augment = False, rir_root=""):
        self.root_dir = root_dir  # Path to the LA/DF folder

        protocol_file = os.path.join(self.root_dir, protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file, sep=" ", header=None)

        self.rec_dir = os.path.join(self.root_dir, "flac")

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


class ASVspoof2021LADataset_pair(ASVspoof2021_base):
    """
    Dataset class for ASVspoof2021 LA that provides pairs of genuine and tested speech for differential-based detection.
    """

    def __init__(self, root_dir, protocol_file_name, augment = False, rir_root=""):
        super().__init__(root_dir, protocol_file_name, augment, rir_root)

        headers = ["SPEAKER_ID", "AUDIO_FILE_NAME", "-", "-", "MODIF", "KEY", "-", "VARIANT"]
        self.protocol_df.columns = headers
        self.protocol_df = self.protocol_df[self.protocol_df["VARIANT"] == "eval"]

    def __getitem__(self, idx):
        """
        Returns tuples of the form (test_audio_file_name, gt_waveform, test_waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]  # Get the speaker ID

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, f"{test_audio_file_name}.flac")
        test_waveform, _ = load(test_audio_name)  # Load the tested speech

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == "bonafide" else 1  # 0 for genuine speech, 1 for spoofing speech

        # Get the genuine speech of the same speaker for differentiation
        speaker_recordings_df = self.protocol_df[
            (self.protocol_df["SPEAKER_ID"] == speaker_id) & (self.protocol_df["KEY"] == "bonafide")
        ]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        # Get a random genuine speech of the speaker using sample()
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, f"{gt_audio_file_name}.flac")
        gt_waveform, _ = load(gt_audio_name)  # Load the genuine speech

        # print(f"Loaded GT:{gt_audio_name} and TEST:{test_audio_name}")
        return test_audio_file_name, gt_waveform, test_waveform, label


class ASVspoof2021LADataset_single(ASVspoof2021_base):
    """
    Dataset class for ASVspoof2021 LA that provides single speech samples for "normal" detection.
    """

    def __init__(self, root_dir, protocol_file_name, augment = False, rir_root=""):
        super().__init__(root_dir, protocol_file_name, augment, rir_root)

        headers = ["SPEAKER_ID", "AUDIO_FILE_NAME", "-", "-", "MODIF", "KEY", "-", "VARIANT"]
        self.protocol_df.columns = headers
        self.protocol_df = self.protocol_df[self.protocol_df["VARIANT"] == "eval"]

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        audio_name = os.path.join(self.rec_dir, f"{audio_file_name}.flac")
        waveform, _ = load(audio_name)

        # 0 for genuine speech, 1 for spoofing speech
        label = 0 if self.protocol_df.loc[idx, "KEY"] == "bonafide" else 1

        return audio_file_name, waveform, label


class ASVspoof2021DFDataset_pair(ASVspoof2021_base):
    """
    Dataset class for ASVspoof2021 DF that provides pairs of genuine and tested speech for differential-based detection.
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["progress", "eval"] = "eval",
        local: bool = False,
        augment=False,
        rir_root="",
    ):
        super().__init__(root_dir, protocol_file_name, augment)

        headers = [
            "SPEAKER_ID",
            "AUDIO_FILE_NAME",
            "-",
            "SOURCE",
            "MODIF",
            "KEY",
            "-",
            "VARIANT",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
        self.protocol_df.columns = headers
        self.protocol_df = self.protocol_df[self.protocol_df["VARIANT"] == variant]

        if local:  # Needed because locally there is only a subset of the eval set
            # Keep recordings that are only present in the flac directory
            # get the list of files in the flac directory
            print("Loading the list of files in the flac directory")
            present_files = os.listdir(self.rec_dir)
            # remove the .flac extension
            present_files = [f.split(".")[0] for f in present_files]

            self.protocol_df = self.protocol_df[self.protocol_df["AUDIO_FILE_NAME"].isin(present_files)]

            print(f"Using {len(self.protocol_df)} local recordings from DF21 eval set.")

        self.protocol_df.reset_index(drop=True, inplace=True)

    def __getitem__(self, idx):
        """
        Returns tuples of the form (test_audio_file_name, gt_waveform, test_waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]  # Get the speaker ID

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, f"{test_audio_file_name}.flac")
        test_waveform, _ = load(test_audio_name)  # Load the tested speech

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == "bonafide" else 1  # 0 for genuine speech, 1 for spoofing speech

        # Get the genuine speech of the same speaker for differentiation
        speaker_recordings_df = self.protocol_df[
            (self.protocol_df["SPEAKER_ID"] == speaker_id) & (self.protocol_df["KEY"] == "bonafide")
        ]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        # Get a random genuine speech of the speaker using sample()
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, f"{gt_audio_file_name}.flac")
        gt_waveform, _ = load(gt_audio_name)  # Load the genuine speech

        # print(f"Loaded GT:{gt_audio_name} and TEST:{test_audio_name}")
        return test_audio_file_name, gt_waveform, test_waveform, label


class ASVspoof2021DFDataset_single(ASVspoof2021_base):
    """
    Dataset class for ASVspoof2021 DF that provides single speech samples for "normal" detection.
    """

    def __init__(
        self,
        root_dir,
        protocol_file_name,
        variant: Literal["progress", "eval"] = "eval",
        local: bool = False,
        augment=False,
        rir_root="",
    ):
        super().__init__(root_dir, protocol_file_name, augment)

        headers = [
            "SPEAKER_ID",
            "AUDIO_FILE_NAME",
            "-",
            "SOURCE",
            "MODIF",
            "KEY",
            "-",
            "VARIANT",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
        self.protocol_df.columns = headers
        self.protocol_df = self.protocol_df[self.protocol_df["VARIANT"] == variant]

        if local:  # Needed because locally there is only a subset of the eval set
            # Keep recordings that are only present in the flac directory
            # get the list of files in the flac directory
            print("Loading the list of files in the flac directory")
            present_files = os.listdir(self.rec_dir)
            # remove the .flac extension
            present_files = [f.split(".")[0] for f in present_files]

            self.protocol_df = self.protocol_df[self.protocol_df["AUDIO_FILE_NAME"].isin(present_files)]

            print(f"Using {len(self.protocol_df)} local recordings from DF21 eval set.")

        self.protocol_df.reset_index(drop=True, inplace=True)

    def __getitem__(self, idx):
        """
        Returns tuples of the form (audio_file_name, waveform, label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        audio_name = os.path.join(self.rec_dir, f"{audio_file_name}.flac")
        waveform, _ = load(audio_name)

        # 0 for genuine speech, 1 for spoofing speech
        label = 0 if self.protocol_df.loc[idx, "KEY"] == "bonafide" else 1

        return audio_file_name, waveform, label
