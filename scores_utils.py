#!/usr/bin/env python

from itertools import combinations
import json
from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm

from trainers.utils import calculate_EER

from config import local_config


def draw_score_distribution(c="FFConcat1", ep=15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"./scores/DF21/{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)

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
    plt.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5", ymax=0.8, alpha=0.7)
    plt.xlabel("Scores")
    plt.ylabel("Relative frequency of bonafide/spoofed")
    plt.title(f"Distribution of scores: {c}")
    plt.legend(loc="upper center")
    # plt.xlim(0, 1)
    plt.savefig(f"./scores/{c}_{ep}_scores.png")


def split_scores_VC_TTS(c="FFConcat1", ep=15):
    # Load the scores
    scores_headers = ["AUDIO_FILE_NAME", "SCORE", "LABEL"]
    scores_df = pd.read_csv(f"./scores/DF21/{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)
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

    asvspoof_bonafide_df = protocol_df[
        (protocol_df["KEY"] == "bonafide") & (protocol_df["SOURCE"].str.contains("asvspoof"))
    ].reset_index(drop=True)

    tts_systems = ["A01", "A02", "A03", "A04", "A07", "A08", "A09", "A10", "A11", "A12", "A16"]
    tts_subset = protocol_df[protocol_df["MODIF"].isin(tts_systems)].reset_index(drop=True)
    tts_subset = pd.concat([tts_subset, asvspoof_bonafide_df], axis=0)

    vc_systems = ["A05", "A06", "A17", "A18", "A19"]
    asvspoof_vc_subset = protocol_df[protocol_df["MODIF"].isin(vc_systems)].reset_index(drop=True)
    vcc_subset = protocol_df[protocol_df["SOURCE"].str.contains("vcc")].reset_index(drop=True)
    vc_subset = pd.concat([asvspoof_vc_subset, vcc_subset, asvspoof_bonafide_df], axis=0)

    for subset, subset_df in zip(["TTS", "VC"], [tts_subset, vc_subset]):
        eer = calculate_EER(c, subset_df["LABEL"], subset_df["SCORE"], False, f"{subset}_{c}")
        print(f"EER for {subset}: {eer*100}%")


def fusion_NN():
    # Code working but not doing what I want
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = torch.nn.Linear(6, 1, device=d)

    # Load the scores
    all_scores_df = pd.DataFrame()
    for c, ep in [
        ("FFDiff", 20),
        ("FFDiffAbs", 15),
        ("FFDiffQuadratic", 15),
        ("FFConcat1", 15),
        ("FFConcat3", 10),
        ("FFLSTM", 10),
    ]:
        print(f"Loading scores for {c}_{ep}")
        scores_headers = ["AUDIO_FILE_NAME", f"SCORE_{c}", "LABEL"]
        scores_df = pd.read_csv(f"./scores/DF21/{c}_{c}_{ep}.pt_scores.txt", sep=",", names=scores_headers)
        if all_scores_df.empty:
            all_scores_df.insert(0, "AUDIO_FILE_NAME", scores_df["AUDIO_FILE_NAME"])
            all_scores_df.insert(1, "LABEL", scores_df["LABEL"])
            all_scores_df.insert(2, f"SCORE_{c}", scores_df[f"SCORE_{c}"])
        else:
            all_scores_df = all_scores_df.merge(scores_df, on=["AUDIO_FILE_NAME", "LABEL"])
    # print(all_scores_df.iloc[:, 2:])

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(layer.parameters())
    for epoch in range(20):
        batch_size = 512
        num_scores = all_scores_df.shape[0]
        losses = []
        predictions = []
        for i in tqdm(range(0, num_scores, batch_size)):
            optimizer.zero_grad()
            batch_scores = all_scores_df.iloc[i : i + batch_size, 2:].to_numpy()
            batch_labels = all_scores_df.iloc[i : i + batch_size, 1].to_numpy()
            inputs = torch.tensor(batch_scores, dtype=torch.float32).to(d)
            labels = torch.tensor(batch_labels, dtype=torch.float32).to(d)
            outputs = layer(inputs).squeeze()
            pred = torch.abs(torch.round(outputs))
            predictions.extend(pred)
            loss_value = loss(outputs, labels)
            loss_value.backward()
            losses.append(loss_value.item())
            optimizer.step()
        accuracy = torch.mean(
            (torch.tensor(predictions) == torch.tensor(all_scores_df["LABEL"])).float()
        ).item()
        print(f"Epoch {epoch}: Loss: {torch.mean(torch.tensor(losses))}, Acc: {accuracy}")


def fusion_scores(dataset: Literal["DF21", "InTheWild"]):
    dfs = []

    for file in os.listdir(f"./scores/{dataset}"):
        if ".json" in file or "fusion" in file:
            continue  # Skip the fusion scores

        df = pd.read_csv(
            f"./scores/{dataset}/{file}", header=None, names=["file", f'score_{file.split("_")[1]}', "label"]
        )
        dfs.append(df)

    final_df = pd.concat(dfs, axis=0).groupby(["file", "label"]).first().reset_index()
    if dataset == "DF21":
        final_df = final_df.drop(columns=["score_FF"])  # Only pair-input systems

    scores = [
        "score_FFConcat1",
        "score_FFConcat2",
        "score_FFConcat3",
        "score_FFDiff",
        "score_FFDiffAbs",
        "score_FFDiffQuadratic",
        "score_FFLSTM",
        "score_FFLSTM2",
    ]
    comb = []
    for i in range(2, len(scores) + 1):
        comb.extend(combinations(scores, i))

    score_dict = {}
    for combination in tqdm(comb):
        name = f" + ".join(combination)
        mean_score = final_df[list(combination)].mean(axis=1)
        max_score = final_df[list(combination)].max(axis=1)
        min_score = final_df[list(combination)].min(axis=1)
        sqrt_score = final_df[list(combination)].apply(lambda x: x.prod() ** (1 / len(combination)), axis=1)

        mean_eer = calculate_EER(name, final_df["label"], mean_score, False, "")
        max_eer = calculate_EER(name, final_df["label"], max_score, False, "")
        min_eer = calculate_EER(name, final_df["label"], min_score, False, "")
        sqrt_eer = calculate_EER(name, final_df["label"], sqrt_score, False, "")

        score_dict[name] = {"mean": mean_eer, "max": max_eer, "min": min_eer, "sqrt": sqrt_eer}

    json.dump(score_dict, open(f"./scores/{dataset}/fusion_scores.json", "w"))


def fusion_scores_from_json(
    dataset: Literal["DF21", "InTheWild"], number: Literal["all", "oneplusone"] = "all"
):
    scores = json.load(open(f"./scores/{dataset}/fusion_scores.json", "r"))

    if number == "oneplusone":
        doubles = {key: scores[key] for key in scores if len(key.split(" + ")) == 2}
        scores = {
            key: doubles[key]
            for key in doubles
            if (key.count("FFDiff") == 1 and (key.count("FFConcat") == 1 or key.count("FFLSTM") == 1))
        }

    for fusion in ["mean", "max", "min", "sqrt"]:
        best_fusion = min(scores, key=lambda x: scores[x][fusion])
        print(f"Best {fusion} fusion: {best_fusion}, EER: {scores[best_fusion][fusion]*100}%")


if __name__ == "__main__":
    print("##### DF21 #####")
    fusion_scores("DF21")
    fusion_scores_from_json("DF21", "oneplusone")
    fusion_scores_from_json("DF21", "all")
    print("##### InTheWild #####")
    fusion_scores("InTheWild")
    fusion_scores_from_json("InTheWild", "oneplusone")
    fusion_scores_from_json("InTheWild", "all")
