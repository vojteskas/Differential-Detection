#!/usr/bin/env python3

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio import load # type: ignore

class MLDFDataset(Dataset):
    def __init__(self):
        self.root = "/mnt/strade/istanek/datasets/BP_xtrnov01/datasets" # HARDCODED!!!!

        self.df_en = pd.read_csv(f"{self.root}/metadata/metadata_EN.csv", sep=" ")
        # self.df_en["wav_file"] = "dataset_EN/" + self.df_en["wav_file"].astype(str)
        self.df_de = pd.read_csv(f"{self.root}/metadata/metadata_DE.csv", sep=" ")
        # self.df_de["wav_file"] = "dataset_DE/" + self.df_de["wav_file"].astype(str)
        self.df_fr = pd.read_csv(f"{self.root}/metadata/metadata_FR.csv", sep=" ")
        # self.df_fr["wav_file"] = "dataset_FR/" + self.df_fr["wav_file"].astype(str)
        self.df_es = pd.read_csv(f"{self.root}/metadata/metadata_ES.csv", sep=" ")
        # self.df_es["wav_file"] = "dataset_ES/" + self.df_es["wav_file"].astype(str)
        self.df_it = pd.read_csv(f"{self.root}/metadata/metadata_IT.csv", sep=" ")
        # self.df_it["wav_file"] = "dataset_IT/" + self.df_it["wav_file"].astype(str)

        self.df = pd.concat([self.df_en, self.df_de, self.df_fr, self.df_es, self.df_it])
        self.df = self.df[self.df["group"] == "test"] # only test data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wf, sr = load(self.root + "/" + row["wav_file"])
        if wf.shape[1] > 64600:
            wf = wf[:, :64600]  # Trim to 64600 samples
        elif wf.shape[1] < 64600:
            pad_length = 64600 - wf.shape[1]
            wf = torch.nn.functional.pad(wf, (0, pad_length))  # Pad to 64600 samples
        # returns name, waveform, tool (bonafide or name of synthesizer), M/F gender
        return row["wav_file"], wf.squeeze(), row["tool"], row["gender"]


mldf_dataset = MLDFDataset()
mldf_dataloader = DataLoader(mldf_dataset, batch_size=64, shuffle=False)

if __name__ == "__main__":
    for i, data in enumerate(mldf_dataloader):
        print(data)
        if i == 10:
            break
