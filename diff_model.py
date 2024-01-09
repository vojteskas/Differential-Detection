import torch
import torch.nn as nn
import torch.nn.functional as F

from w2v_model import SSLModel

# TODO: Add dropout?
# TODO: Migrate to PyTorch Lightning?
# TODO: Migrate to newer PyTorch version and use torch.hub to load the build-in 
#       wav2vec model instead of using the custom one

class DiffModel(nn.Module):
    def __init__(self, device):
        super(DiffModel, self).__init__()

        # self.wav2vec = SSLModel(device)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        ).to(device)

        self.device = device
        return

    # Without fine-tuning wav2vec, expects embeddings of shape (batch_size, 1024, 1)
    def forward(self, emb_gt, emb_test):
        emb_gt = emb_gt.to(self.device)
        emb_test = emb_test.to(self.device)
        diff = emb_gt - emb_test

        # print(f"Diff shape: {diff.shape}")

        out = self.classifier(diff.squeeze(-1))
        prob = F.softmax(out, dim=1)

        return out, prob
    
    # With fine-tuning wav2vec, expects collated raw audio inputs
    # def forward(self, input_data_ground_truth, input_data_tested):
        # emb_gt = self.wav2vec.extract_feat(input_data_ground_truth)
        # emb_test = self.wav2vec.extract_feat(input_data_tested)

        # emb_gt = torch.mean(emb_gt, dim=1)
        # emb_test = torch.mean(emb_test, dim=1)
    
        # diff = emb_gt - emb_test

        # print(f"Diff shape: {diff.shape}")

        # out = self.classifier(diff.squeeze(-1))
        # prob = F.softmax(out, dim=1)

        # return out, prob
