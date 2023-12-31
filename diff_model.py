import torch
import torch.nn as nn

from w2v_model import SSLModel

# TODO: Update for batch training after adding collate function to dataset (dont forget to add batch normalization)


class DiffModel(nn.Module):
    def __init__(self, device):
        super(DiffModel, self).__init__()

        self.wav2vec = SSLModel(device)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.device = device
        return

    def forward(self, input_data_ground_truth, input_data_tested):
        print(f"GT: {input_data_ground_truth.shape}, TEST: {input_data_tested.shape}")
        batch_size = input_data_ground_truth.shape[0]

        emb_gt = self.wav2vec.extract_feat(input_data_ground_truth)
        emb_test = self.wav2vec.extract_feat(input_data_tested)

        emb_gt = torch.mean(emb_gt, dim=1)
        emb_test = torch.mean(emb_test, dim=1)

        diff = emb_gt - emb_test

        out = self.classifier(diff)

        return out
