import os
from typing import Literal
import pandas as pd
from torch.utils.data import IterableDataset
from torchaudio import load


class Morphing_base(IterableDataset):
    """
    Base class for the Morphing examples from https://github.com/itsuki8914/Voice-morphing-RelGAN/tree/master/result_examples
    This class should not be used directly, but rather subclassed.

    param root_dir: Path to the root folder
    param protocol_file_name: Name of the protocol file to use
    """

    def __init__(self, root_dir: str, protocol_file_name: str):
        self.root_dir = root_dir

        # Load the protocol file
        protocol_file = os.path.join(root_dir, protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file, sep=" ")

    def __len__(self):
        return len(self.protocol_df)

    def __iter__(self):
        raise NotImplementedError("Child classes should implement the __getitem__ method")


class MorphingDataset_single(Morphing_base):
    def __init__(self, root_dir: str, protocol_file_name: str, variant: Literal["eval"] = "eval"):
        super().__init__(root_dir, protocol_file_name)

    def __iter__(self):
        for index, row in self.protocol_df.iterrows():
            test_audio = os.path.join(self.root_dir, row["FILE"])
            test_waveform, _ = load(test_audio)
            yield row["FILE"], test_waveform, 0 if row["LABEL"] == "bonafide" else 1


class MorphingDataset_pair(Morphing_base):
    def __init__(self, root_dir: str, protocol_file_name: str, variant: Literal["eval"] = "eval"):
        super().__init__(root_dir, protocol_file_name)

    def __iter__(self):
        for index, row in self.protocol_df.iterrows():
            test_audio = os.path.join(self.root_dir, row["FILE"])
            test_waveform, _ = load(test_audio)

            label = 0 if row["LABEL"] == "bonafide" else 1

            # Get the genuine speech
            speaker1_df = self.protocol_df[
                (self.protocol_df["SPEAKER_ID1"] == row["SPEAKER_ID1"])
                & (self.protocol_df["LABEL"] == "bonafide")
            ]
            gt_audio1 = speaker1_df.sample(n=1).iloc[0]["FILE"]
            gt_waveform1, _ = load(os.path.join(self.root_dir, gt_audio1))
            yield row["FILE"], gt_waveform1, test_waveform, label

            if row["LABEL"] == "morph":
                # Get the second speaker's genuine speech
                speaker2_df = self.protocol_df[
                    (self.protocol_df["SPEAKER_ID1"] == int(row["SPEAKER_ID2"]))
                    & (self.protocol_df["LABEL"] == "bonafide")
                ]
                gt_audio2 = speaker2_df.sample(n=1).iloc[0]["FILE"]
                gt_waveform2, _ = load(os.path.join(self.root_dir, gt_audio2))
                yield row["FILE"], gt_waveform2, test_waveform, label
