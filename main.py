#!/usr/bin/env python3

import librosa
import torch
import numpy as np
from datasets import ASVspoof2019Dataset
from torch.utils.data import DataLoader

from diff_model import DiffModel


def init_test():
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    print(f'Using device: {d}')
    model = SSLModel(device=d)

    # load audio
    audio, _ = librosa.load('./real.flac', sr=None)
    audio_real = torch.from_numpy(audio[np.newaxis, :]).to(d)
    audio, _ = librosa.load('./real2.flac', sr=None)
    audio_real2 = torch.from_numpy(audio[np.newaxis, :]).to(d)
    audio, _ = librosa.load('./fake.flac', sr=None)
    audio_fake = torch.from_numpy(audio[np.newaxis, :]).to(d)

    print('### Audio shape:', audio_real.shape, audio_real2.shape, audio_fake.shape)

    # extract feature
    feat_real = model.extract_feat(audio_real)
    feat_real2 = model.extract_feat(audio_real2)
    feat_fake = model.extract_feat(audio_fake)

    # compute mean along the second axis and squeeze the first dimension
    feat_real = torch.mean(feat_real, dim=1).squeeze(0)
    feat_real2 = torch.mean(feat_real2, dim=1).squeeze(0)
    feat_fake = torch.mean(feat_fake, dim=1).squeeze(0)

    # print
    print('### Real:', feat_real.shape, '\n', feat_real)
    print('### Real2:', feat_real2.shape, '\n', feat_real2)
    print('### Fake:', feat_fake.shape, '\n', feat_fake)

    # compute differences
    diff_real = feat_real - feat_real2
    diff_fake = feat_real - feat_fake

    # print differences
    print('### Difference between real recordings:', diff_real)
    print('### Difference between real and fake:', diff_fake)

    # compute Euclidean distances
    dist_real = torch.dist(feat_real, feat_real2)
    dist_fake = torch.dist(feat_real, feat_fake)

    # print distances
    print('### Euclidean distance between real recordings:', dist_real.item())
    print('### Euclidean distance between real and fake:', dist_fake.item())

def batch_test():
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {d}')
    
    # load model
    model = DiffModel(device=d)
    model.load_state_dict(torch.load('./diffmodel.pt'))
    dataset = ASVspoof2019Dataset(
        root_dir="/mnt/e/VUT/Deepfakes/Datasets/LA",
        protocol_file_name="ASVspoof2019.LA.cm.train.trn.txt",
    )
    dataloader = DataLoader(dataset, shuffle=True)

    model.to(d)
    model.train()
    batch = next(iter(dataloader))
    output = model(batch[0], batch[1])

    print(output.shape, output)


if __name__ == '__main__':
    batch_test()
