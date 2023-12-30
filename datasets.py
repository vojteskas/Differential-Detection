from typing import Literal
import torch
from torch.utils.data import Dataset
from torchaudio import load
import os
import pandas as pd

# TODO: Add collate function so that the dataloader can return a batch of data

class ASVspoof2019Dataset(Dataset):
    def __init__(self, root_dir, protocol_file_name, variant: Literal['train', 'dev', 'eval'] = 'train'):
        self.root_dir = root_dir # Path to the LA folder

        protocol_file = os.path.join(self.root_dir, 'ASVspoof2019_LA_cm_protocols', protocol_file_name)
        self.protocol_df = pd.read_csv(protocol_file, sep=" ", header=None)
        self.protocol_df.columns = ["SPEAKER_ID", "AUDIO_FILE_NAME", "SYSTEM_ID", "-", "KEY"]

        self.rec_dir = os.path.join(self.root_dir, 'ASVspoof2019_LA_' + variant, 'flac')

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        speaker_id = self.protocol_df.loc[idx, "SPEAKER_ID"]

        test_audio_file_name = self.protocol_df.loc[idx, "AUDIO_FILE_NAME"]
        test_audio_name = os.path.join(self.rec_dir, test_audio_file_name + '.flac')
        test_waveform, _ = load(test_audio_name)

        label = self.protocol_df.loc[idx, "KEY"]
        label = 0 if label == 'bonafide' else 1  # 0 for genuine speech, 1 for spoofing speech

        speaker_recordings_df = self.protocol_df[self.protocol_df["SPEAKER_ID"] == speaker_id]
        if speaker_recordings_df.empty:
            raise Exception(f"Speaker {speaker_id} genuine speech not found in protocol file")
        gt_audio_file_name = speaker_recordings_df.sample(n=1).iloc[0]["AUDIO_FILE_NAME"]
        gt_audio_name = os.path.join(self.rec_dir, gt_audio_file_name + '.flac')
        gt_waveform, _ = load(gt_audio_name)

        print(f"Loaded GT:{gt_audio_name} and TEST:{test_audio_name}")
        return gt_waveform, test_waveform, label

# Usage
# dataset = ASVSpoofDataset(root_dir='/path/to/ASVSpoof2019/dataset')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
