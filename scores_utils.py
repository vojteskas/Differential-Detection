from matplotlib.pylab import f
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from trainers.utils import calculate_EER

def draw_score_distribution(c = "FFConcat1", ep = 15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"../{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)

    # Filter the scores based on label
    bf_hist, bf_edges = np.histogram(scores_df[scores_df["LABEL"] == 0]["SCORE"], bins=15)
    sp_hist, sp_edges = np.histogram(scores_df[scores_df["LABEL"] == 1]["SCORE"], bins=15)
    bf_freq = bf_hist / np.sum(bf_hist)
    sp_freq = sp_hist / np.sum(sp_hist)
    bf_width = np.diff(bf_edges)
    sp_width = np.diff(sp_edges)
    plt.bar(bf_edges[:-1], bf_freq, width=bf_width, alpha=0.5, label="Bonafide", color="green", edgecolor="darkgreen", linewidth=1.5, align='edge')
    plt.bar(sp_edges[:-1], sp_freq, width=sp_width, alpha=0.5, label="Spoofed", color="red", edgecolor="darkred", linewidth=1.5, align='edge')
    plt.xlabel("Scores")
    plt.ylabel("Relative frequency of bonafide/spoofed")
    plt.title(f"Distribution of scores: {c}")
    plt.legend()
    plt.show()

def split_scores_VC_TTS(c = "FFConcat1", ep = 15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"../{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)
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
    protocol_df = pd.read_csv("../DF21_protocol.txt", sep=" ")
    protocol_df.columns = df21_headers
    protocol_df = protocol_df.merge(scores_df, on="AUDIO_FILE_NAME")
    eer = calculate_EER(c, protocol_df["LABEL"], protocol_df["SCORE"], False, f"DF21_{c}")
    print(f"EER for DF21: {eer*100}%")

    for subset in ["vcc2018", "vcc2020", "asvspoof", "vcc"]:
        protocol_subset = protocol_df[protocol_df["SOURCE"].str.contains(subset)].reset_index(drop=True)
        eer = calculate_EER(c, protocol_subset["LABEL"], protocol_subset["SCORE"], False, f"{subset}_{c}")
        print(f"EER for {subset}: {eer*100}%")


if __name__ == "__main__":
    # draw_score_distribution(c="FFDiff", ep=20)
    # draw_score_distribution(c="FFConcat1", ep=15)
    split_scores_VC_TTS(c="FFConcat1", ep=15)
