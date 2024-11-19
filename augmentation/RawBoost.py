#!/usr/bin/env python3

"""
RawBoost data augmentation for audio data. Created by Hemlata Tak et al., 2021.
Copied from https://github.com/TakHemlata/RawBoost-antispoofing
"""

import numpy as np
from scipy import signal
import copy
import torch

# RawBoost data augmentation for audio data
class Args:
    def __init__(self):
        # RawBoost default parameters
        self.N_f = 5
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20
        self.P = 10
        self.g_sd = 2
        self.SNRmin = 10
        self.SNRmax = 40


args = Args()


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y


def normWav(x, always):
    try:
        if always or (abs(x).max()) > 1:
            x /= abs(x).max()
        return x
    except Exception as e:
        print(f"x: {x}")
        print(f"x.shape: {x.shape}")
        print(f"abs(x): {abs(x)}")
        print(f"abs(x).max(): {abs(x).max()}")
        raise e


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    b = 1
    for _ in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)

    G = randRange(minG, maxG, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filterFIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(np.array(y).shape[0] - N / 2)]
    return y


# Linear and non-linear convolutive noise
def LnL_convolutive_noise(
    x,
    N_f,
    nBands,
    minF,
    maxF,
    minBW,
    maxBW,
    minCoeff,
    maxCoeff,
    minG,
    maxG,
    minBiasLinNonLin,
    maxBiasLinNonLin,
    fs,
):
    # print("LnL_convolutive_noise")
    # print(f"x.shape[0]: {x.shape[0]}")
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG - minBiasLinNonLin
            maxG = maxG - maxBiasLinNonLin
        b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
        filtered = filterFIR(np.power(x, (i + 1)), b)
        # print(f"filtered.shape[0]: {filtered.shape[0]}")
        y += filtered
    y = y - np.mean(y)
    y = normWav(y, 0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    beta = randRange(0, P, 0)

    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    p = np.random.permutation(x_len)[:n]
    f_r = np.multiply(((2 * np.random.rand(p.shape[0])) - 1), ((2 * np.random.rand(p.shape[0])) - 1))
    r = g_sd * x[p] * f_r
    y[p] = torch.tensor(x[p] + r, dtype=torch.float32)
    y = normWav(y, 0)
    return y


# Stationary signal independent noise
def SSI_additive_noise(
    x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * SNR)
    x = x + noise
    return x


def process_Rawboost_feature(feature, sr, algo=5):  # algo = 5 is default LnL + ISD

    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature,
            args.SNRmin,
            args.SNRmax,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            sr,
        )

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f,
            args.nBands,
            args.minF,
            args.maxF,
            args.minBW,
            args.maxBW,
            args.minCoeff,
            args.maxCoeff,
            args.minG,
            args.maxG,
            args.minBiasLinNonLin,
            args.maxBiasLinNonLin,
            sr,
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = np.add(feature1, feature2)
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return torch.Tensor(feature)
