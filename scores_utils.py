#!/usr/bin/env python

from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from trainers.utils import calculate_EER

from config import local_config


def draw_score_distribution(c="FFConcat1", ep=15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"./scores/{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)

    # Filter the scores based on label
    bf_hist, bf_edges = np.histogram(scores_df[scores_df["LABEL"] == 0]["SCORE"], bins=15)
    sp_hist, sp_edges = np.histogram(scores_df[scores_df["LABEL"] == 1]["SCORE"], bins=15)
    bf_freq = bf_hist / np.sum(bf_hist)
    sp_freq = sp_hist / np.sum(sp_hist)
    bf_width = np.diff(bf_edges)
    sp_width = np.diff(sp_edges)
    plt.figure(figsize=(8, 5))
    plt.bar(
        bf_edges[:-1],
        bf_freq,
        width=(bf_width + sp_width) / 2,
        alpha=0.5,
        label="Bonafide",
        color="green",
        edgecolor="darkgreen",
        linewidth=1.5,
        align="edge",
    )
    plt.bar(
        sp_edges[:-1],
        sp_freq,
        width=(bf_width + sp_width) / 2,
        alpha=0.5,
        label="Spoofed",
        color="red",
        edgecolor="darkred",
        linewidth=1.5,
        align="edge",
    )
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold 0.5', ymax=0.8, alpha=0.7)
    plt.xlabel("Scores")
    plt.ylabel("Relative frequency of bonafide/spoofed")
    plt.title(f"Distribution of scores: {c}")
    plt.legend(loc='upper center')
    # plt.xlim(0, 1)
    plt.savefig(f"./scores/{c}_{ep}_scores.png")


def split_scores_VC_TTS(c="FFConcat1", ep=15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"./scores/{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)
    scores_df["SCORE"] = scores_df["SCORE"].astype(float)

    # Load DF21 protocol
    df21_headers = [
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
    protocol_df = pd.read_csv(
        f'{local_config["data_dir"]}{local_config["asvspoof2021df"]["eval_subdir"]}/{local_config["asvspoof2021df"]["eval_protocol"]}',
        sep=" ",
    )
    protocol_df.columns = df21_headers
    protocol_df = protocol_df.merge(scores_df, on="AUDIO_FILE_NAME")
    eer = calculate_EER(c, protocol_df["LABEL"], protocol_df["SCORE"], False, f"DF21_{c}")
    print(f"EER for DF21: {eer*100}%")

    for subset in ["vcc2018", "vcc2020", "asvspoof", "vcc"]:
        protocol_subset = protocol_df[protocol_df["SOURCE"].str.contains(subset)].reset_index(drop=True)
        eer = calculate_EER(c, protocol_subset["LABEL"], protocol_subset["SCORE"], False, f"{subset}_{c}")
        print(f"EER for {subset}: {eer*100}%")


if __name__ == "__main__":
    for c, ep in [
        ("FFDiff", 20),
        ("FFDiffAbs", 15),
        ("FFDiffQuadratic", 15),
        ("FFConcat1", 15),
        ("FFConcat3", 10),
        ("FFLSTM", 10),
    ]:
        print(f"Classifier: {c}")
        draw_score_distribution(c, ep)
        split_scores_VC_TTS(c, ep)
        print("\n")
